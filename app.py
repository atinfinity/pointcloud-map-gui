"""Open3D GUI application: load point clouds, visualize with a height filter,
and export a ROS2 map_server-compatible occupancy grid (PGM + YAML).
"""
import os
import threading
import traceback

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from colorize import GRAY_OUT_COLOR, height_colormap_colors
from ground_grid import build_ground_grid_lines
from map_preview import downsample_to_thumbnail
from map_writer import export_map
from occupancy_grid import compute_occupancy_grid
from pointcloud_io import load_point_cloud

IN_RANGE_NAME = "pointcloud_in_range"
OUT_OF_RANGE_NAME = "pointcloud_out_of_range"
AXES_NAME = "coordinate_frame"
GRID_NAME = "ground_grid"
DEFAULT_RESOLUTION = 0.05
OUT_OF_RANGE_ALPHA = 0.15
MAP_PREVIEW_MAX_DIM = 200


class MainWindow:
    def __init__(self):
        self.window = gui.Application.instance.create_window(
            "Point Cloud -> Occupancy Grid Exporter", 1280, 800
        )
        em = self.window.theme.font_size

        self.pcd = None  # currently loaded o3d.geometry.PointCloud
        self.points = None  # (N,3) numpy view of pcd.points
        self.height_colors = None  # (N,3) height colormap, fixed per loaded cloud

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
        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))

        panel.add_child(gui.Label("Point Cloud"))
        load_button = gui.Button("Load Point Cloud (.pcd/.ply)...")
        load_button.set_on_clicked(self._on_load_clicked)
        panel.add_child(load_button)

        self.info_label = gui.Label("No point cloud loaded.")
        panel.add_child(self.info_label)

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
        export_button = gui.Button("Export Occupancy Grid...")
        export_button.set_on_clicked(self._on_export_clicked)
        panel.add_child(export_button)

        panel.add_fixed(em)
        self.status_label = gui.Label("")
        panel.add_child(self.status_label)

        panel.add_stretch()
        panel.add_child(gui.Label("Occupancy Grid Preview"))
        placeholder = np.full((MAP_PREVIEW_MAX_DIM, MAP_PREVIEW_MAX_DIM, 3), 60, dtype=np.uint8)
        self.map_preview_widget = gui.ImageWidget(o3d.geometry.Image(placeholder))
        panel.add_child(self.map_preview_widget)

        self._set_height_controls_enabled(False)

        # --- layout ---
        self.panel = panel
        self.window.add_child(self.scene_widget)
        self.window.add_child(panel)
        self.window.set_on_layout(self._on_layout)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 22 * layout_context.theme.font_size
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)

    def _set_height_controls_enabled(self, enabled):
        for w in (
            self.min_height_edit,
            self.max_height_edit,
            self.min_height_slider,
            self.max_height_slider,
            self.resolution_edit,
        ):
            w.enabled = enabled

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
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self._on_load_error(str(e))
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_load_success(self, path, pcd):
        self.pcd = pcd
        # np.asarray() on an Open3D vector is a zero-copy view into its buffer,
        # and pcd.colors gets reassigned to a brand new buffer on every
        # height-filter update (whose address the allocator can reuse) -- so
        # this snapshot must be a real copy, taken before any of that happens.
        self.points = np.asarray(pcd.points).copy()

        z_min = float(self.points[:, 2].min())
        z_max = float(self.points[:, 2].max())
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

        n = self.points.shape[0]
        bbox = self.pcd.get_axis_aligned_bounding_box()
        self.info_label.text = (
            f"{os.path.basename(path)}\n"
            f"{n:,} points\n"
            f"X: [{bbox.min_bound[0]:.2f}, {bbox.max_bound[0]:.2f}]\n"
            f"Y: [{bbox.min_bound[1]:.2f}, {bbox.max_bound[1]:.2f}]\n"
            f"Z: [{bbox.min_bound[2]:.2f}, {bbox.max_bound[2]:.2f}]"
        )
        self.status_label.text = "Loaded."

        # Height colors only depend on each point's Z relative to the full
        # cloud's Z range, which is fixed at load time -- computing this
        # once and slicing it per height-filter update is much cheaper than
        # recomputing the colormap on every slider tick.
        self.height_colors = height_colormap_colors(self.points)

        self._refresh_point_cloud_geometry()
        self._refresh_axes(bbox)
        self._refresh_ground_grid(bbox)

        self._set_initial_camera(bbox)
        self.window.set_needs_layout()

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

        in_pcd = o3d.geometry.PointCloud()
        in_pcd.points = o3d.utility.Vector3dVector(self.points[in_range_mask])
        in_pcd.colors = o3d.utility.Vector3dVector(self.height_colors[in_range_mask])

        out_points = self.points[~in_range_mask]
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(out_points)
        out_pcd.colors = o3d.utility.Vector3dVector(
            np.tile(GRAY_OUT_COLOR, (out_points.shape[0], 1))
        )

        for name, geo, material in (
            (IN_RANGE_NAME, in_pcd, self.point_material),
            (OUT_OF_RANGE_NAME, out_pcd, self.out_of_range_material),
        ):
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.remove_geometry(name)
            self.scene_widget.scene.add_geometry(name, geo, material)

        self._refresh_map_preview()

    def _refresh_map_preview(self):
        if self.pcd is None:
            return
        min_h, max_h = self._current_height_range()
        resolution = self.resolution_edit.double_value
        try:
            result = compute_occupancy_grid(self.points, min_h, max_h, resolution)
        except ValueError:
            return
        thumbnail = downsample_to_thumbnail(result.grid, MAP_PREVIEW_MAX_DIM)
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
            min_h, max_h = self._current_height_range()
            resolution = self.resolution_edit.double_value
            result = compute_occupancy_grid(self.points, min_h, max_h, resolution)
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
