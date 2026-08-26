"""Open3D GUI application: load point clouds, visualize with a height filter,
and export a ROS2 map_server-compatible occupancy grid (PGM + YAML).
"""
import os
import sys
import threading
import time
import traceback

import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from colorize import fade_to_background, height_colormap_colors
from display_lod import select_display_indices
from ground_grid import build_ground_grid_lines
import ground_removal
import noise_removal
from map_preview import downsample_to_thumbnail
from map_writer import export_map
from noise_worker import NoiseWorker
from occupancy_grid import compute_occupancy_grid, remove_small_occupied_blobs
from pointcloud_io import load_point_cloud

POINTS_NAME = "pointcloud"
NOISE_COLOR = np.array([1.0, 0.3, 0.3])
AXES_NAME = "coordinate_frame"
GRID_NAME = "ground_grid"
DEFAULT_RESOLUTION = 0.05
DEFAULT_MIN_BLOB_CELLS = 0  # map cleanup off until the user opts in
SCENE_BACKGROUND = np.array([0.35, 0.35, 0.35])
# De-emphasised points are drawn faded rather than translucent -- see
# colorize.fade_to_background for why.
NOISE_ALPHA = 0.15
# Ground sits a little stronger than that, and in its own hue: it is now the
# only category still drawn de-emphasised, and a neutral grey that faint reads
# as "those points were deleted" rather than "those points are the floor".
GROUND_ALPHA = 0.30
GROUND_TINT = np.array([0.35, 0.65, 1.0])
GROUND_COLOR = fade_to_background(GROUND_TINT, GROUND_ALPHA, SCENE_BACKGROUND)
NOISE_DISPLAY_COLOR = fade_to_background(NOISE_COLOR, NOISE_ALPHA, SCENE_BACKGROUND)
# Drawing every point of a multi-million-point cloud costs far more than it
# shows: the view is thinned to this many points unless the user raises it.
# The occupancy grid and the export always use the full cloud.
DEFAULT_MAX_DISPLAY_POINTS = 1_000_000
# Upper bound and pre-layout fallback for the preview thumbnail. The real
# size is whatever _on_layout gives the widget: handing it a larger image and
# letting it scale down would undo the obstacle-preserving reduction in
# map_preview and drop thin walls again.
MAP_PREVIEW_MAX_DIM = 800
MAP_PREVIEW_ALPHA = 128  # 0-255 (~0.5); the overlay lets the point cloud show through
PREVIEW_SCENE_WIDTH_FRACTION = 0.2  # overlay side <= 20% of the 3D view width
PREVIEW_SCENE_HEIGHT_FRACTION = 0.25  # ... and <= 25% of its height
GROUND_METHODS = list(ground_removal.DEFAULT_PARAMS)  # combobox order
NOISE_METHODS = list(noise_removal.DEFAULT_PARAMS)


def reindex_ground_mask(ground_mask, previous_removed, new_removed, total):
    """Move a ground classification onto a point set with different points
    removed from it, or None if it cannot be carried over.

    Changing the noise filter only adds or removes points; every point that
    survives keeps whatever it was classified as. Dropping the classification
    instead leaves the ground drawn in full colour until a fresh estimate
    arrives -- measured at ~75 ms, which on a cloud that is 40% floor is a
    visible flash. Points that reappear come back unclassified, until the
    estimate this is accompanied by refines them.

    `previous_removed` and `new_removed` are noise masks over all `total`
    points, or None for "nothing removed". `ground_mask` is over whatever
    survived `previous_removed`.
    """
    if ground_mask is None:
        return None
    expected = total if previous_removed is None else int((~previous_removed).sum())
    if ground_mask.shape[0] != expected:
        return None  # stale, and mis-indexing it would be worse than starting over
    if previous_removed is None:
        lifted = ground_mask
    else:
        lifted = np.zeros(total, dtype=bool)
        lifted[~previous_removed] = ground_mask
    return lifted if new_removed is None else lifted[~new_removed]


class MethodSection:
    """Checkbox + method combobox + one parameter block per method (only the
    selected block is visible) + result label. Shared by noise and ground
    removal, which have identical `DEFAULT_PARAMS`-driven UIs."""

    def __init__(self, panel, em, checkbox_text, methods, module, on_method_changed, on_setting_changed):
        self.methods = methods
        self.module = module
        self.checkbox = gui.Checkbox(checkbox_text)
        self.checkbox.set_on_checked(on_setting_changed)
        panel.add_child(self.checkbox)
        self.combo = gui.Combobox()
        for method in methods:
            self.combo.add_item(module.METHOD_LABELS[method])
        self.combo.set_on_selection_changed(on_method_changed)
        panel.add_child(self.combo)
        self.param_edits = {}
        self.param_blocks = {}
        for method in methods:
            block = gui.Vert(0.25 * em)
            edits = {}
            for name, (default, lo, hi) in module.DEFAULT_PARAMS[method].items():
                is_int = isinstance(default, int)
                edit = gui.NumberEdit(gui.NumberEdit.INT if is_int else gui.NumberEdit.DOUBLE)
                edit.set_limits(lo, hi)
                if is_int:
                    edit.int_value = default
                else:
                    edit.double_value = default
                edit.set_on_value_changed(on_setting_changed)
                row = gui.Horiz(0.5 * em)
                row.add_child(gui.Label(f"{name}:"))
                row.add_stretch()
                row.add_child(edit)
                block.add_child(row)
                edits[name] = edit
            self.param_edits[method] = edits
            self.param_blocks[method] = block
            panel.add_child(block)
        self.info_label = gui.Label("")
        panel.add_child(self.info_label)
        self.show_param_block(methods[0])

    @property
    def enabled(self):
        return self.checkbox.checked

    @property
    def selected_method(self):
        return self.methods[self.combo.selected_index]

    def show_param_block(self, method):
        for name, block in self.param_blocks.items():
            block.visible = name == method

    def current_params(self):
        method = self.selected_method
        params = self.module.default_params(method)
        for name, edit in self.param_edits[method].items():
            params[name] = edit.int_value if isinstance(params[name], int) else edit.double_value
        return params

    def set_enabled(self, enabled):
        self.checkbox.enabled = enabled
        self.combo.enabled = enabled
        for edits in self.param_edits.values():
            for edit in edits.values():
                edit.enabled = enabled


class MainWindow:
    def __init__(self, noise_worker=None):
        # Without a started worker this falls back to a thread, which is
        # correct but freezes the GUI while Open3D holds the GIL.
        self.noise_worker = noise_worker if noise_worker is not None else NoiseWorker()
        self.window = gui.Application.instance.create_window(
            "Point Cloud -> Occupancy Grid Exporter", 1280, 800
        )
        em = self.window.theme.font_size

        self.pcd = None  # currently loaded o3d.geometry.PointCloud
        self.all_points = None  # (M,3) every loaded point
        self.points = None  # (N,3) active points = all_points minus detected noise
        self.noise_points = None  # (M-N,3) removed noise points, for display only
        self.noise_mask = None  # (M,) bool, None when nothing is being removed
        # A noise result in flight is about to change which points exist, so
        # ground estimation waits for it rather than producing a mask for
        # points that are on their way out.
        self._noise_pending = False
        self._ground_deferred = False
        self.height_colors = None  # (N,3) height colormap over the active points
        self.ground_mask = None  # (N,) bool ground points, None when removal is off
        # Generation counters discard stale results from superseded worker threads.
        self._noise_job_id = 0
        self._ground_job_id = 0
        self._preview_job_id = 0

        # --- drawn subset ---
        # Positions are uploaded once per point-set change; height-filter and
        # ground-removal updates then rewrite colors in place and push only
        # those, which is what keeps big clouds interactive.
        self._display_positions = None  # (D,3) float32, active points then noise
        # What is actually uploaded. Points outside the height filter get NaN
        # coordinates here, which the renderer culls -- the only way to hide a
        # point without splitting the geometry, which is what made this slow.
        self._display_draw_positions = None
        self._display_colors = None  # (D,3) float32, the buffer handed to the GPU
        self._display_base_colors = None  # (Da,3) float32 height colors, unfaded
        self._display_zs = None  # (Da,) float32
        self._display_active_index = None  # indices into self.points, None = all
        self._display_active_count = 0  # Da; noise points occupy [Da:]

        # Slider drags fire far faster than a rebuild can run. Callbacks only
        # raise these, and the tick handler collapses a burst into one update.
        self._colors_dirty = False
        self._preview_dirty = False
        self._preview_side = MAP_PREVIEW_MAX_DIM  # replaced by the first layout

        # --- 3D scene widget ---
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background(list(SCENE_BACKGROUND) + [1.0])
        self.point_material = rendering.MaterialRecord()
        self.point_material.shader = "defaultUnlit"
        self.point_material.point_size = 2.0
        self.axes_material = rendering.MaterialRecord()
        self.axes_material.shader = "defaultLit"
        self.grid_material = rendering.MaterialRecord()
        self.grid_material.shader = "unlitLine"
        self.grid_material.line_width = 1.5
        self.grid_material.base_color = [0.55, 0.55, 0.55, 1.0]

        # --- control panel ---
        # Controls live in a scrollable column so a short window scrolls
        # instead of squeezing widgets on top of each other; the map preview
        # is overlaid on the 3D view (top-right) by _on_layout so it can be
        # large without competing with the controls for space.
        panel = gui.ScrollableVert(0.5 * em, gui.Margins(em, em, em, em))

        panel.add_child(gui.Label("Point Cloud"))
        load_button = gui.Button("Load Point Cloud (.pcd/.ply)...")
        load_button.set_on_clicked(self._on_load_clicked)
        panel.add_child(load_button)

        self.info_label = gui.Label("No point cloud loaded.")
        panel.add_child(self.info_label)

        max_display_row = gui.Horiz(0.5 * em)
        max_display_row.add_child(gui.Label("Max display points:"))
        max_display_row.add_stretch()
        self.max_display_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.max_display_edit.set_limits(0, 1_000_000_000)
        self.max_display_edit.int_value = DEFAULT_MAX_DISPLAY_POINTS
        self.max_display_edit.set_on_value_changed(self._on_max_display_changed)
        max_display_row.add_child(self.max_display_edit)
        panel.add_child(max_display_row)

        panel.add_fixed(em)
        noise_group = gui.CollapsableVert("Noise Removal", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self.noise_section = MethodSection(
            noise_group, em, "Remove isolated points", NOISE_METHODS, noise_removal,
            self._on_noise_method_changed, self._on_noise_setting_changed,
        )
        panel.add_child(noise_group)

        panel.add_fixed(em)
        ground_group = gui.CollapsableVert("Ground Removal", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self.ground_section = MethodSection(
            ground_group, em, "Remove ground points", GROUND_METHODS, ground_removal,
            self._on_ground_method_changed, self._on_ground_setting_changed,
        )
        panel.add_child(ground_group)

        panel.add_fixed(em)
        panel.add_child(gui.Label("Height Filter"))

        self.min_height_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.max_height_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.min_height_slider = gui.Slider(gui.Slider.DOUBLE)
        self.max_height_slider = gui.Slider(gui.Slider.DOUBLE)

        self.min_height_edit.set_on_value_changed(self._on_min_height_edit)
        self.max_height_edit.set_on_value_changed(self._on_max_height_edit)
        self.min_height_slider.set_on_value_changed(self._on_min_height_slider)
        self.max_height_slider.set_on_value_changed(self._on_max_height_slider)

        min_row = gui.Horiz(0.5 * em)
        min_row.add_child(gui.Label("Min:"))
        min_row.add_child(self.min_height_edit)
        panel.add_child(min_row)
        panel.add_child(self.min_height_slider)

        max_row = gui.Horiz(0.5 * em)
        max_row.add_child(gui.Label("Max:"))
        max_row.add_child(self.max_height_edit)
        panel.add_child(max_row)
        panel.add_child(self.max_height_slider)

        panel.add_fixed(em)
        panel.add_child(gui.Label("Occupancy Grid Resolution (m/cell)"))
        self.resolution_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.resolution_edit.double_value = DEFAULT_RESOLUTION
        self.resolution_edit.set_limits(0.001, 5.0)
        self.resolution_edit.set_on_value_changed(self._on_resolution_changed)
        panel.add_child(self.resolution_edit)

        panel.add_fixed(em)
        panel.add_child(gui.Label("Map Cleanup: min occupied blob (cells)"))
        self.min_blob_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.min_blob_edit.int_value = DEFAULT_MIN_BLOB_CELLS
        self.min_blob_edit.set_limits(0, 10000)
        self.min_blob_edit.set_on_value_changed(self._on_resolution_changed)
        panel.add_child(self.min_blob_edit)

        panel.add_fixed(em)
        export_button = gui.Button("Export Occupancy Grid...")
        export_button.set_on_clicked(self._on_export_clicked)
        panel.add_child(export_button)

        panel.add_fixed(em)
        self.status_label = gui.Label("")
        panel.add_child(self.status_label)

        # The preview caption and image are laid out by hand in _on_layout
        # as an overlay on the scene widget (children added later draw on
        # top; a Vert would only grant the ImageWidget its pixel size).
        self.preview_label = gui.Label("Occupancy Grid Preview")
        placeholder = np.full((MAP_PREVIEW_MAX_DIM, MAP_PREVIEW_MAX_DIM, 4), 60, dtype=np.uint8)
        placeholder[..., 3] = MAP_PREVIEW_ALPHA
        self.map_preview_widget = gui.ImageWidget(o3d.geometry.Image(placeholder))

        self._set_height_controls_enabled(False)

        # --- layout ---
        self.panel = panel
        self.window.add_child(self.scene_widget)
        self.window.add_child(panel)
        self.window.add_child(self.preview_label)
        self.window.add_child(self.map_preview_widget)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_tick_event(self._on_tick)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = layout_context.theme.font_size
        panel_width = 22 * em
        panel_x = r.get_right() - panel_width

        scene_width = r.width - panel_width
        self.panel.frame = gui.Rect(panel_x, r.y, panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, scene_width, r.height)

        # Preview overlay: caption + square image in the top-right corner of
        # the 3D view, sized relative to the view so it stays readable at any
        # window size without hiding most of the point cloud.
        label_height = self.preview_label.calc_preferred_size(
            layout_context, gui.Widget.Constraints()
        ).height
        image_side = int(
            min(
                scene_width * PREVIEW_SCENE_WIDTH_FRACTION,
                r.height * PREVIEW_SCENE_HEIGHT_FRACTION - label_height,
            )
        )
        image_side = max(image_side, 1)
        if image_side != self._preview_side:
            # Rebuild the thumbnail at the size it will actually be shown at.
            self._preview_side = image_side
            self._preview_dirty = True
        image_x = r.x + scene_width - em - image_side
        label_y = r.y + em
        image_y = label_y + label_height + int(0.25 * em)
        self.preview_label.frame = gui.Rect(image_x, label_y, image_side, label_height)
        self.map_preview_widget.frame = gui.Rect(image_x, image_y, image_side, image_side)

    def _set_height_controls_enabled(self, enabled):
        for w in (
            self.min_height_edit,
            self.max_height_edit,
            self.min_height_slider,
            self.max_height_slider,
            self.resolution_edit,
            self.min_blob_edit,
        ):
            w.enabled = enabled
        self.noise_section.set_enabled(enabled)
        self.ground_section.set_enabled(enabled)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _on_load_clicked(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select point cloud file", self.window.theme)
        dlg.add_filter(".pcd .ply", "Point cloud files (*.pcd, *.ply)")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_load_file_selected)
        self.window.show_dialog(dlg)

    def _on_load_file_selected(self, path):
        self.window.close_dialog()
        self.status_label.text = f"Loading {os.path.basename(path)}..."
        self.window.set_needs_layout()

        def worker():
            try:
                pcd = load_point_cloud(path)
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._on_load_success(path, pcd)
                )
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                # Python unbinds `e` at the end of the except block, so it
                # must be captured into a plain variable before being used
                # inside a lambda that runs later on the main thread.
                error_message = str(e)
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._on_load_error(error_message)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_load_success(self, path, pcd):
        self.pcd = pcd
        self.ground_mask = None
        self.loaded_name = os.path.basename(path)
        # np.asarray() on an Open3D vector is a zero-copy view into its buffer,
        # and pcd.colors gets reassigned to a brand new buffer on every
        # height-filter update (whose address the allocator can reuse) -- so
        # this snapshot must be a real copy, taken before any of that happens.
        self.all_points = np.asarray(pcd.points).copy()
        self.points = self.all_points
        self.noise_points = np.empty((0, 3))
        self.noise_mask = None

        z_min = float(self.all_points[:, 2].min())
        z_max = float(self.all_points[:, 2].max())
        if z_min == z_max:
            z_max = z_min + 1e-3

        self.min_height_slider.set_limits(z_min, z_max)
        self.max_height_slider.set_limits(z_min, z_max)
        self.min_height_edit.set_limits(z_min, z_max)
        self.max_height_edit.set_limits(z_min, z_max)

        self.min_height_slider.double_value = z_min
        self.max_height_slider.double_value = z_max
        self.min_height_edit.double_value = z_min
        self.max_height_edit.double_value = z_max

        self._set_height_controls_enabled(True)
        self.status_label.text = "Loaded."

        bbox = self.pcd.get_axis_aligned_bounding_box()
        self._set_initial_camera(bbox)
        # Noise first: it marks a result as pending, which is what makes the
        # ground pass inside _apply_active_points hold off. The other way round
        # the ground mask is computed for points the noise pass is about to
        # remove, shown, and then replaced -- which reads as a flicker.
        self._request_noise_update()
        self._apply_active_points()

    def _active_bbox(self):
        return o3d.geometry.AxisAlignedBoundingBox(self.points.min(axis=0), self.points.max(axis=0))

    def _apply_active_points(self, ground_mask=None):
        """Refresh everything derived from the active (noise-filtered) point
        set: info text, height colors, axes/grid, scene, then ground mask.

        `ground_mask` is the classification carried over from the previous
        point set where one could be, so the ground keeps its tint until the
        fresh estimate lands instead of flashing back to full colour.
        """
        bbox = self._active_bbox()
        # Height colors only depend on each point's Z relative to the active
        # cloud's Z range, which is fixed until the noise filter changes --
        # computing this once and slicing it per height-filter update is much
        # cheaper than recomputing the colormap on every slider tick.
        self.height_colors = height_colormap_colors(self.points)
        self.ground_mask = ground_mask
        self._rebuild_display_geometry()
        self._refresh_axes(bbox)
        self._refresh_ground_grid(bbox)
        self.window.set_needs_layout()
        self._request_ground_update()

    def _set_initial_camera(self, bbox):
        center = bbox.get_center()
        extent = max(bbox.get_max_extent(), 1e-3)
        distance = extent * 1.5
        # Eye offset only in Y/Z (no X component): forward and up then both
        # lie in the Y-Z plane, which forces the screen-right vector to be
        # pure world +X -- i.e. X reads as horizontal, pointing right.
        eye = center + np.array([0.0, -distance, distance], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.scene_widget.setup_camera(60, bbox, center)
        self.scene_widget.look_at(center, eye, up)

    def _on_load_error(self, message):
        self.status_label.text = f"Error loading file: {message}"
        self.window.set_needs_layout()

    def _rebuild_display_geometry(self):
        """Re-pick the drawn subset and upload it.

        Only needed when the point set itself changes (load, noise filter,
        display budget). Height-filter and ground-removal changes go through
        _update_display_buffers, which rewrites what is drawn without
        rebuilding it.
        """
        if self.pcd is None:
            return
        budget = self.max_display_edit.int_value
        active_index = select_display_indices(self.points, budget)
        active = self.points if active_index is None else self.points[active_index]
        base = self.height_colors if active_index is None else self.height_colors[active_index]

        # Noise gets its own slice of the budget rather than competing for the
        # same one, so a mostly-noise cloud cannot crowd out the points the
        # user is actually filtering.
        noise_index = select_display_indices(self.noise_points, max(budget // 4, 1) if budget else 0)
        noise = self.noise_points if noise_index is None else self.noise_points[noise_index]

        self._display_active_index = active_index
        self._display_active_count = active.shape[0]
        self._display_positions = np.ascontiguousarray(np.vstack([active, noise]), dtype=np.float32)
        self._display_base_colors = np.ascontiguousarray(base, dtype=np.float32)
        self._display_zs = np.ascontiguousarray(active[:, 2], dtype=np.float32)
        self._display_draw_positions = self._display_positions.copy()
        self._display_colors = np.empty_like(self._display_positions)
        # Noise never changes color and is never hidden by the height filter,
        # so both of its buffers are written once here and the per-tick rewrite
        # below only ever touches the active slice.
        self._display_colors[self._display_active_count:] = NOISE_DISPLAY_COLOR
        self._write_display_buffers()

        scene = self.scene_widget.scene
        if scene.has_geometry(POINTS_NAME):
            scene.remove_geometry(POINTS_NAME)
        if self._display_positions.shape[0]:
            # No downsampled copy: this class does its own thinning, and
            # Open3D's copy would keep whatever colors it was built with.
            scene.add_geometry(POINTS_NAME, self._display_tensor(), self.point_material, False)
        self._refresh_info_label()
        self._preview_dirty = True

    def _display_tensor(self):
        """Tensor point cloud over the display buffers. Tensors created this
        way share the numpy memory, so this is a wrapper, not a copy."""
        tensor = o3d.t.geometry.PointCloud(o3c.Tensor.from_numpy(self._display_draw_positions))
        tensor.point.colors = o3c.Tensor.from_numpy(self._display_colors)
        return tensor

    def _write_display_buffers(self):
        """Fill the drawn colors and positions for the active slice.

        Three outcomes per point: inside the height filter and not ground, so
        drawn in the height colormap; ground, so drawn in its own faded tint
        whatever the filter says; or outside the filter, so not drawn at all.
        Hiding is done by writing NaN coordinates rather than by leaving the
        point out, which would mean rebuilding the geometry on every tick.
        """
        count = self._display_active_count
        if not count:
            return
        min_h, max_h = self._current_height_range()
        visible = (self._display_zs >= min_h) & (self._display_zs <= max_h)

        colors = self._display_colors[:count]
        np.copyto(colors, self._display_base_colors)
        if self.ground_mask is not None:
            index = self._display_active_index
            ground = self.ground_mask if index is None else self.ground_mask[index]
            np.copyto(colors, GROUND_COLOR.astype(np.float32), where=ground[:, None])
            visible |= ground

        positions = self._display_draw_positions[:count]
        np.copyto(positions, self._display_positions[:count])
        np.copyto(positions, np.float32("nan"), where=~visible[:, None])

    def _update_display_buffers(self):
        """Push new colors and positions for an unchanged point set -- the
        cheap path that a height-filter drag takes. Sending positions as well
        as colors measured free: both are one buffer of the same size."""
        if self._display_positions is None or not self._display_positions.shape[0]:
            return
        self._write_display_buffers()
        self.scene_widget.scene.scene.update_geometry(
            POINTS_NAME,
            self._display_tensor(),
            rendering.Scene.UPDATE_COLORS_FLAG | rendering.Scene.UPDATE_POINTS_FLAG,
        )

    def _refresh_info_label(self):
        n = self.points.shape[0]
        noise_line = ""
        if self.noise_points is not None and self.noise_points.shape[0]:
            noise_line = f" ({self.noise_points.shape[0]:,} noise removed)"
        drawn = ""
        if self._display_active_index is not None:
            drawn = f"\nshowing {self._display_active_count:,} (thinned for display)"
        bbox = self._active_bbox()
        self.info_label.text = (
            f"{self.loaded_name}\n"
            f"{n:,} points{noise_line}{drawn}\n"
            f"X: [{bbox.min_bound[0]:.2f}, {bbox.max_bound[0]:.2f}]\n"
            f"Y: [{bbox.min_bound[1]:.2f}, {bbox.max_bound[1]:.2f}]\n"
            f"Z: [{bbox.min_bound[2]:.2f}, {bbox.max_bound[2]:.2f}]"
        )

    def _on_max_display_changed(self, value):
        if self.pcd is None:
            return
        self._rebuild_display_geometry()
        self.window.set_needs_layout()

    # ------------------------------------------------------------------
    # Deferred updates
    # ------------------------------------------------------------------
    def _mark_display_dirty(self):
        self._colors_dirty = True
        self._preview_dirty = True

    def _on_tick(self):
        """Collapse a burst of callbacks into at most one update per frame.

        A slider drag fires far faster than a rebuild can run; without this the
        queued work would keep replaying long after the user let go.
        """
        if not (self._colors_dirty or self._preview_dirty):
            return False
        if self._colors_dirty:
            self._colors_dirty = False
            self._update_display_buffers()
        if self._preview_dirty:
            self._preview_dirty = False
            self._request_map_preview()
        return True

    def _request_map_preview(self):
        """Rebuild the preview on a worker thread: it is a full pass over the
        (undecimated) cloud plus a connected-component cleanup, which is too
        slow to sit on the UI thread for large clouds."""
        if self.pcd is None:
            return
        self._preview_job_id += 1
        job_id = self._preview_job_id
        # Widget state has to be read here -- worker threads must not touch it.
        points = self.points
        ground_mask = self.ground_mask
        min_h, max_h = self._current_height_range()
        resolution = self.resolution_edit.double_value
        min_blob = self.min_blob_edit.int_value
        max_dim = min(self._preview_side, MAP_PREVIEW_MAX_DIM)

        def worker():
            try:
                result = compute_occupancy_grid(
                    points, min_h, max_h, resolution, exclude_mask=ground_mask
                )
                result.grid = remove_small_occupied_blobs(result.grid, min_blob)
                thumbnail = downsample_to_thumbnail(
                    result.grid, max_dim, alpha=MAP_PREVIEW_ALPHA
                )
            except ValueError:
                return  # e.g. an inverted height range; keep the last good preview
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                return
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self._on_map_preview_ready(job_id, thumbnail)
            )

        threading.Thread(target=worker, daemon=True).start()

    def _on_map_preview_ready(self, job_id, thumbnail):
        if job_id != self._preview_job_id:
            return  # superseded by a newer request
        self.map_preview_widget.update_image(o3d.geometry.Image(thumbnail))
        self.window.set_needs_layout()

    def _build_occupancy_grid(self):
        """Full map pipeline on the active points: height filter + ground
        exclusion -> grid -> small-blob cleanup. Raises ValueError on bad
        settings (e.g. inverted height range)."""
        min_h, max_h = self._current_height_range()
        resolution = self.resolution_edit.double_value
        result = compute_occupancy_grid(
            self.points, min_h, max_h, resolution, exclude_mask=self.ground_mask
        )
        result.grid = remove_small_occupied_blobs(result.grid, self.min_blob_edit.int_value)
        return result

    def _on_resolution_changed(self, value):
        self._preview_dirty = True

    def _refresh_axes(self, bbox):
        # Anchor the axes at the point cloud's XY-min corner: that is the
        # occupancy grid's origin on export, so the axes double as a preview
        # of the map origin. Z is pinned to 0 so the axes always sit on the
        # Z=0 ground grid instead of sinking with stray points below floor
        # level (the map origin has no meaningful Z anyway).
        size = max(bbox.get_max_extent() * 0.2, 1e-3)
        axes = self._build_axis_arrows(size)
        axes.translate((bbox.min_bound[0], bbox.min_bound[1], 0.0))
        if self.scene_widget.scene.has_geometry(AXES_NAME):
            self.scene_widget.scene.remove_geometry(AXES_NAME)
        self.scene_widget.scene.add_geometry(AXES_NAME, axes, self.axes_material)

    @staticmethod
    def _build_axis_arrows(size):
        # create_arrow() builds a +Z-pointing arrow at the origin; each axis
        # is a copy rotated onto X/Y/Z, sized thicker than Open3D's default
        # coordinate frame so it reads clearly as an arrow rather than a thin
        # line.
        cylinder_radius = size * 0.05
        cone_radius = size * 0.11
        cylinder_height = size * 0.8
        cone_height = size * 0.2

        specs = [
            ((1.0, 0.0, 0.0), (0.0, np.pi / 2, 0.0)),  # X: red
            ((0.0, 1.0, 0.0), (-np.pi / 2, 0.0, 0.0)),  # Y: green
            ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),  # Z: blue
        ]

        axes = o3d.geometry.TriangleMesh()
        for color, euler in specs:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cylinder_radius,
                cone_radius=cone_radius,
                cylinder_height=cylinder_height,
                cone_height=cone_height,
            )
            arrow.paint_uniform_color(color)
            if any(euler):
                rotation = arrow.get_rotation_matrix_from_xyz(euler)
                arrow.rotate(rotation, center=(0.0, 0.0, 0.0))
            axes += arrow
        axes.compute_vertex_normals()
        return axes

    def _refresh_ground_grid(self, bbox):
        # A wireframe at Z=0 makes the "zero height" plane unambiguous at a
        # glance, independent of wherever the Height Filter sliders happen
        # to be pointed.
        grid = self._build_ground_grid(bbox)
        if self.scene_widget.scene.has_geometry(GRID_NAME):
            self.scene_widget.scene.remove_geometry(GRID_NAME)
        self.scene_widget.scene.add_geometry(GRID_NAME, grid, self.grid_material)

    def _build_ground_grid(self, bbox):
        min_x, min_y = bbox.min_bound[0], bbox.min_bound[1]
        max_x, max_y = bbox.max_bound[0], bbox.max_bound[1]
        points_2d, lines, _ = build_ground_grid_lines(min_x, min_y, max_x, max_y)

        # Sitting the grid at exactly Z=0 z-fights with any point cloud data
        # that also lies at Z=0 (e.g. a flat floor scan), which makes lines
        # flicker in and out. Nudge it a hair below instead.
        z = -max(bbox.get_max_extent(), 1e-3) * 1e-3
        points = [[p[0], p[1], z] for p in points_2d]

        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
        grid.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        return grid

    # ------------------------------------------------------------------
    # Height filter callbacks
    # ------------------------------------------------------------------
    def _on_min_height_edit(self, value):
        clamped = min(value, self.max_height_edit.double_value)
        if clamped != value:
            self.min_height_edit.double_value = clamped
        self.min_height_slider.double_value = clamped
        self._mark_display_dirty()

    def _on_max_height_edit(self, value):
        clamped = max(value, self.min_height_edit.double_value)
        if clamped != value:
            self.max_height_edit.double_value = clamped
        self.max_height_slider.double_value = clamped
        self._mark_display_dirty()

    def _on_min_height_slider(self, value):
        clamped = min(value, self.max_height_slider.double_value)
        if clamped != value:
            self.min_height_slider.double_value = clamped
        self.min_height_edit.double_value = clamped
        self._mark_display_dirty()

    def _on_max_height_slider(self, value):
        clamped = max(value, self.min_height_slider.double_value)
        if clamped != value:
            self.max_height_slider.double_value = clamped
        self.max_height_edit.double_value = clamped
        self._mark_display_dirty()

    def _current_height_range(self):
        return self.min_height_slider.double_value, self.max_height_slider.double_value

    # ------------------------------------------------------------------
    # Noise removal
    # ------------------------------------------------------------------
    def _on_noise_method_changed(self, text, index):
        self.noise_section.show_param_block(self.noise_section.selected_method)
        self.window.set_needs_layout()
        self._request_noise_update()

    def _on_noise_setting_changed(self, value):
        self._request_noise_update()

    def _request_noise_update(self):
        """Recompute the noise mask over all loaded points on a worker thread,
        then rebuild the active point set (which re-runs ground estimation)."""
        if self.pcd is None:
            return
        self._noise_job_id += 1
        job_id = self._noise_job_id
        self._noise_pending = False
        if not self.noise_section.enabled:
            self.noise_section.info_label.text = ""
            self._set_noise_mask(None)
            self._run_deferred_ground_update()
            return

        method = self.noise_section.selected_method
        params = self.noise_section.current_params()
        points = self.all_points
        self.status_label.text = f"Removing noise ({method})..."
        self.window.set_needs_layout()

        # Estimation runs in a separate process: the Open3D methods behind
        # most of these hold the GIL, so a thread would freeze the GUI for as
        # long as they run. See noise_worker.
        def on_done(mask, elapsed, error):
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self._on_noise_result(job_id, method, mask, elapsed, error)
            )

        self._noise_pending = True
        self.noise_worker.submit(points, method, params, on_done)

    def _on_noise_result(self, job_id, method, mask, elapsed, error):
        if job_id != self._noise_job_id:
            return  # superseded by a newer request; a newer one is still pending
        self._noise_pending = False
        if error is not None:
            self.noise_section.info_label.text = f"Noise removal failed:\n{error}"
            self.status_label.text = "Noise removal failed."
            self._set_noise_mask(None)
            self._run_deferred_ground_update()
            return
        if mask.all():
            self.noise_section.info_label.text = "All points classified as noise; ignoring."
            self.status_label.text = "Noise removal rejected."
            self._set_noise_mask(None)
            self._run_deferred_ground_update()
            return
        n_noise = int(mask.sum())
        self.noise_section.info_label.text = (
            f"{method}: {n_noise:,} noise points "
            f"({100.0 * n_noise / mask.shape[0]:.1f}%) in {elapsed:.2f}s"
        )
        self.status_label.text = "Noise removed."
        self._set_noise_mask(mask)
        self._run_deferred_ground_update()

    def _run_deferred_ground_update(self):
        """Issue the ground pass that was held back, unless applying the noise
        mask already rebuilt the point set and issued one itself."""
        if self._ground_deferred:
            self._request_ground_update()

    def _set_noise_mask(self, mask):
        # A mask that removes nothing is the same state as no mask at all, and
        # collapsing it here is what lets the comparison below spot a no-op.
        if mask is not None and not mask.any():
            mask = None
        if self._noise_mask_unchanged(mask):
            # Rebuilding the point set costs a full pass over the cloud and a
            # re-upload -- half a second on a large one, with the GUI frozen
            # for it. Toggling a filter that turns out to remove nothing (or
            # turning one off that never removed anything) must not pay that.
            return
        carried = (
            None
            if self.all_points is None
            else reindex_ground_mask(
                self.ground_mask, self.noise_mask, mask, self.all_points.shape[0]
            )
        )
        self.noise_mask = mask
        if mask is None:
            self.points = self.all_points
            self.noise_points = np.empty((0, 3))
        else:
            self.points = self.all_points[~mask]
            self.noise_points = self.all_points[mask]
        self._apply_active_points(carried)

    def _noise_mask_unchanged(self, mask):
        if self.noise_mask is None or mask is None:
            return self.noise_mask is None and mask is None
        return np.array_equal(self.noise_mask, mask)

    # ------------------------------------------------------------------
    # Ground removal
    # ------------------------------------------------------------------
    def _on_ground_method_changed(self, text, index):
        self.ground_section.show_param_block(self.ground_section.selected_method)
        self.window.set_needs_layout()
        self._request_ground_update()

    def _on_ground_setting_changed(self, value):
        self._request_ground_update()

    def _request_ground_update(self):
        """Recompute the ground mask for the current method/params on a worker
        thread (large clouds can take seconds), then refresh the scene."""
        if self.pcd is None:
            return
        if self._noise_pending:
            # Estimating now would shade points that a pending noise result is
            # about to delete, show that shading, then drop it and shade again
            # -- a visible blink for an answer that was never going to be kept.
            # _on_noise_result asks again once the point set has settled.
            self._ground_deferred = True
            return
        self._ground_deferred = False
        self._ground_job_id += 1
        job_id = self._ground_job_id
        if not self.ground_section.enabled:
            self.ground_mask = None
            self.ground_section.info_label.text = ""
            self._mark_display_dirty()
            return

        method = self.ground_section.selected_method
        params = self.ground_section.current_params()
        points = self.points
        self.status_label.text = f"Estimating ground ({method})..."
        self.window.set_needs_layout()

        def worker():
            try:
                t0 = time.perf_counter()
                mask = ground_removal.estimate_ground_mask(points, method, **params)
                elapsed = time.perf_counter() - t0
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._on_ground_result(job_id, method, mask, elapsed, None)
                )
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                error_message = str(e)
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._on_ground_result(job_id, method, None, 0.0, error_message)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_ground_result(self, job_id, method, mask, elapsed, error):
        if job_id != self._ground_job_id:
            return  # superseded by a newer request
        if error is not None:
            self.ground_mask = None
            self.ground_section.info_label.text = f"Ground removal failed:\n{error}"
            self.status_label.text = "Ground removal failed."
        else:
            self.ground_mask = mask
            n_ground = int(mask.sum())
            self.ground_section.info_label.text = (
                f"{method}: {n_ground:,} ground points "
                f"({100.0 * n_ground / mask.shape[0]:.1f}%) in {elapsed:.2f}s"
            )
            self.status_label.text = "Ground estimated."
        self._mark_display_dirty()
        self.window.set_needs_layout()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _on_export_clicked(self):
        if self.pcd is None:
            self.status_label.text = "Load a point cloud first."
            return

        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Export occupancy grid (basename)", self.window.theme)
        dlg.add_filter(".yaml", "Map YAML (*.yaml)")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_export_path_selected)
        self.window.show_dialog(dlg)

    def _on_export_path_selected(self, path):
        self.window.close_dialog()
        basepath = path
        if basepath.lower().endswith(".yaml"):
            basepath = basepath[: -len(".yaml")]
        elif basepath.lower().endswith(".pgm"):
            basepath = basepath[: -len(".pgm")]

        try:
            result = self._build_occupancy_grid()
            pgm_path, yaml_path = export_map(basepath, result)
            self.status_label.text = f"Exported:\n{pgm_path}\n{yaml_path}"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.status_label.text = f"Export failed: {e}"
        self.window.set_needs_layout()


def run():
    # Start the estimation process before the GUI. Spawning is cheapest while
    # this process is still simple, and a child started afterwards would
    # inherit whatever state Open3D's own threads are in.
    noise_worker = NoiseWorker()
    if not noise_worker.start():
        print(
            f"Noise removal will run in-process and freeze the window while it does: "
            f"{noise_worker.failure}",
            file=sys.stderr,
        )
    gui.Application.instance.initialize()
    MainWindow(noise_worker)
    try:
        gui.Application.instance.run()
    finally:
        noise_worker.close()
