"""Open3D GUI application: load point clouds, visualize with a height filter,
and export a ROS2 map_server-compatible occupancy grid (PGM + YAML).
"""
import os
import threading
import time
import traceback

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from colorize import GRAY_OUT_COLOR, height_colormap_colors
from ground_grid import build_ground_grid_lines
import ground_removal
import noise_removal
from map_preview import downsample_to_thumbnail
from map_writer import export_map
from occupancy_grid import compute_occupancy_grid, remove_small_occupied_blobs
from pointcloud_io import load_point_cloud

IN_RANGE_NAME = "pointcloud_in_range"
OUT_OF_RANGE_NAME = "pointcloud_out_of_range"
NOISE_NAME = "pointcloud_noise"
NOISE_COLOR = np.array([1.0, 0.3, 0.3])
AXES_NAME = "coordinate_frame"
GRID_NAME = "ground_grid"
DEFAULT_RESOLUTION = 0.05
DEFAULT_MIN_BLOB_CELLS = 0  # map cleanup off until the user opts in
OUT_OF_RANGE_ALPHA = 0.15
MAP_PREVIEW_MAX_DIM = 800
MAP_PREVIEW_ALPHA = 128  # 0-255 (~0.5); the overlay lets the point cloud show through
PREVIEW_SCENE_WIDTH_FRACTION = 0.2  # overlay side <= 20% of the 3D view width
PREVIEW_SCENE_HEIGHT_FRACTION = 0.25  # ... and <= 25% of its height
GROUND_METHODS = list(ground_removal.DEFAULT_PARAMS)  # combobox order
NOISE_METHODS = list(noise_removal.DEFAULT_PARAMS)


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
    def __init__(self):
        self.window = gui.Application.instance.create_window(
            "Point Cloud -> Occupancy Grid Exporter", 1280, 800
        )
        em = self.window.theme.font_size

        self.pcd = None  # currently loaded o3d.geometry.PointCloud
        self.all_points = None  # (M,3) every loaded point
        self.points = None  # (N,3) active points = all_points minus detected noise
        self.noise_points = None  # (M-N,3) removed noise points, for display only
        self.height_colors = None  # (N,3) height colormap over the active points
        self.ground_mask = None  # (N,) bool ground points, None when removal is off
        # Generation counters discard stale results from superseded worker threads.
        self._noise_job_id = 0
        self._ground_job_id = 0

        # --- 3D scene widget ---
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.35, 0.35, 0.35, 1.0])
        self.point_material = rendering.MaterialRecord()
        self.point_material.shader = "defaultUnlit"
        self.point_material.point_size = 2.0
        self.out_of_range_material = rendering.MaterialRecord()
        self.out_of_range_material.shader = "defaultLitTransparency"
        self.out_of_range_material.point_size = 2.0
        self.out_of_range_material.has_alpha = True
        self.out_of_range_material.base_color = [1.0, 1.0, 1.0, OUT_OF_RANGE_ALPHA]
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
        self._apply_active_points()
        self._request_noise_update()

    def _active_bbox(self):
        return o3d.geometry.AxisAlignedBoundingBox(self.points.min(axis=0), self.points.max(axis=0))

    def _apply_active_points(self):
        """Refresh everything derived from the active (noise-filtered) point
        set: info text, height colors, axes/grid, scene, then ground mask."""
        n = self.points.shape[0]
        bbox = self._active_bbox()
        noise_line = ""
        if self.noise_points is not None and self.noise_points.shape[0]:
            noise_line = f" ({self.noise_points.shape[0]:,} noise removed)"
        self.info_label.text = (
            f"{self.loaded_name}\n"
            f"{n:,} points{noise_line}\n"
            f"X: [{bbox.min_bound[0]:.2f}, {bbox.max_bound[0]:.2f}]\n"
            f"Y: [{bbox.min_bound[1]:.2f}, {bbox.max_bound[1]:.2f}]\n"
            f"Z: [{bbox.min_bound[2]:.2f}, {bbox.max_bound[2]:.2f}]"
        )
        # Height colors only depend on each point's Z relative to the active
        # cloud's Z range, which is fixed until the noise filter changes --
        # computing this once and slicing it per height-filter update is much
        # cheaper than recomputing the colormap on every slider tick.
        self.height_colors = height_colormap_colors(self.points)
        self.ground_mask = None
        self._refresh_point_cloud_geometry()
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

    def _refresh_point_cloud_geometry(self):
        if self.pcd is None:
            return
        # Points outside the height filter are rendered as a separate,
        # translucent geometry (its own material/alpha) so they fade into
        # the background instead of competing with the in-range points.
        min_h, max_h = self._current_height_range()
        zs = self.points[:, 2]
        in_range_mask = (zs >= min_h) & (zs <= max_h)
        if self.ground_mask is not None:
            in_range_mask &= ~self.ground_mask

        in_pcd = o3d.geometry.PointCloud()
        in_pcd.points = o3d.utility.Vector3dVector(self.points[in_range_mask])
        in_pcd.colors = o3d.utility.Vector3dVector(self.height_colors[in_range_mask])

        out_points = self.points[~in_range_mask]
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(out_points)
        out_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(GRAY_OUT_COLOR, (out_points.shape[0], 1))
        )

        noise_pcd = o3d.geometry.PointCloud()
        noise_pcd.points = o3d.utility.Vector3dVector(self.noise_points)
        noise_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(NOISE_COLOR, (self.noise_points.shape[0], 1))
        )

        for name, geo, material in (
            (IN_RANGE_NAME, in_pcd, self.point_material),
            (OUT_OF_RANGE_NAME, out_pcd, self.out_of_range_material),
            (NOISE_NAME, noise_pcd, self.out_of_range_material),
        ):
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.remove_geometry(name)
            self.scene_widget.scene.add_geometry(name, geo, material)

        self._refresh_map_preview()

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

    def _refresh_map_preview(self):
        if self.pcd is None:
            return
        try:
            result = self._build_occupancy_grid()
        except ValueError:
            return
        thumbnail = downsample_to_thumbnail(result.grid, MAP_PREVIEW_MAX_DIM, alpha=MAP_PREVIEW_ALPHA)
        self.map_preview_widget.update_image(o3d.geometry.Image(thumbnail))
        self.window.set_needs_layout()

    def _on_resolution_changed(self, value):
        self._refresh_map_preview()

    def _refresh_axes(self, bbox):
        # Anchor the axes at the point cloud's XY-min/Z-min corner: this is
        # the same point used as the occupancy grid's origin on export, so
        # the axes double as a preview of the map origin.
        size = max(bbox.get_max_extent() * 0.2, 1e-3)
        axes = self._build_axis_arrows(size)
        axes.translate(bbox.min_bound)
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
        self._refresh_point_cloud_geometry()

    def _on_max_height_edit(self, value):
        clamped = max(value, self.min_height_edit.double_value)
        if clamped != value:
            self.max_height_edit.double_value = clamped
        self.max_height_slider.double_value = clamped
        self._refresh_point_cloud_geometry()

    def _on_min_height_slider(self, value):
        clamped = min(value, self.max_height_slider.double_value)
        if clamped != value:
            self.min_height_slider.double_value = clamped
        self.min_height_edit.double_value = clamped
        self._refresh_point_cloud_geometry()

    def _on_max_height_slider(self, value):
        clamped = max(value, self.min_height_slider.double_value)
        if clamped != value:
            self.max_height_slider.double_value = clamped
        self.max_height_edit.double_value = clamped
        self._refresh_point_cloud_geometry()

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
        if not self.noise_section.enabled:
            self.noise_section.info_label.text = ""
            self._set_noise_mask(None)
            return

        method = self.noise_section.selected_method
        params = self.noise_section.current_params()
        points = self.all_points
        self.status_label.text = f"Removing noise ({method})..."
        self.window.set_needs_layout()

        def worker():
            try:
                t0 = time.perf_counter()
                mask = noise_removal.estimate_noise_mask(points, method, **params)
                elapsed = time.perf_counter() - t0
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._on_noise_result(job_id, method, mask, elapsed, None)
                )
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                error_message = str(e)
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._on_noise_result(job_id, method, None, 0.0, error_message)
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_noise_result(self, job_id, method, mask, elapsed, error):
        if job_id != self._noise_job_id:
            return  # superseded by a newer request
        if error is not None:
            self.noise_section.info_label.text = f"Noise removal failed:\n{error}"
            self.status_label.text = "Noise removal failed."
            self._set_noise_mask(None)
            return
        if mask.all():
            self.noise_section.info_label.text = "All points classified as noise; ignoring."
            self.status_label.text = "Noise removal rejected."
            self._set_noise_mask(None)
            return
        n_noise = int(mask.sum())
        self.noise_section.info_label.text = (
            f"{method}: {n_noise:,} noise points "
            f"({100.0 * n_noise / mask.shape[0]:.1f}%) in {elapsed:.2f}s"
        )
        self.status_label.text = "Noise removed."
        self._set_noise_mask(mask)

    def _set_noise_mask(self, mask):
        if mask is None:
            self.points = self.all_points
            self.noise_points = np.empty((0, 3))
        else:
            self.points = self.all_points[~mask]
            self.noise_points = self.all_points[mask]
        self._apply_active_points()

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
        self._ground_job_id += 1
        job_id = self._ground_job_id
        if not self.ground_section.enabled:
            self.ground_mask = None
            self.ground_section.info_label.text = ""
            self._refresh_point_cloud_geometry()
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
        self._refresh_point_cloud_geometry()
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
    gui.Application.instance.initialize()
    MainWindow()
    gui.Application.instance.run()
