import sys
import os
import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import cKDTree

os.environ["QT_API"] = "pyqt5"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QFileDialog,
    QStatusBar, QSplitter, QGroupBox, QFrame, QHeaderView,
    QMessageBox, QProgressBar, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette

import pyvista as pv
from pyvistaqt import BackgroundPlotter

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MONSON_RADIUS = 101.6          # mm
CUSP_DETECT_RADIUS = 2.0       # neighborhood radius for local maxima
OCCLUSAL_PERCENTILE = 85       # top N% of Z-values = occlusal surface

APP_TITLE  = "DOPA — Digital Occlusal Plane Analyzer"
APP_VERSION = "v2.0"

# App-directory resolution (works as .py and PyInstaller .exe)
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# DARK PALETTE
# ─────────────────────────────────────────────
DARK_BG      = "#1e1e2e"
DARK_SURFACE = "#2a2a3e"
DARK_BORDER  = "#3a3a5a"
ACCENT_BLUE  = "#4f8ef7"
ACCENT_GREEN = "#50fa7b"
ACCENT_RED   = "#ff5555"
ACCENT_AMBER = "#ffb347"
TEXT_PRIMARY = "#cdd6f4"
TEXT_MUTED   = "#888aaa"

BTN_STYLE = """
QPushButton {{
    background-color: {bg};
    color: {fg};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {hover};
    border-color: {fg};
}}
QPushButton:disabled {{
    background-color: #2a2a3e;
    color: #555577;
    border-color: #333355;
}}
"""

def make_btn(text, bg, fg, hover, border=None):
    border = border or bg
    btn = QPushButton(text)
    btn.setStyleSheet(BTN_STYLE.format(bg=bg, fg=fg, hover=hover, border=border))
    btn.setMinimumHeight(36)
    return btn


# ─────────────────────────────────────────────
# BACKGROUND WORKER — cusp detection
# ─────────────────────────────────────────────
class CuspDetector(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray)
    error    = pyqtSignal(str)

    def __init__(self, points):
        super().__init__()
        self.points = points

    def run(self):
        try:
            pts = self.points
            self.progress.emit("Extracting occlusal surface…")
            z_vals = pts[:, 2]
            z_thr  = np.percentile(z_vals, OCCLUSAL_PERCENTILE)
            occ    = pts[z_vals > z_thr]

            self.progress.emit(f"Running KDTree cusp detection on {len(occ):,} points…")
            kdtree = cKDTree(occ)
            neighbor_idx = kdtree.query_ball_point(occ, r=CUSP_DETECT_RADIUS)

            cusps = []
            for i, p in enumerate(occ):
                nbrs = occ[neighbor_idx[i]]
                if p[2] == np.max(nbrs[:, 2]):
                    cusps.append(p)

            cusp_arr = np.array(cusps) if cusps else np.empty((0, 3))
            self.progress.emit(f"Detected {len(cusp_arr)} cusp tips.")
            self.finished.emit(cusp_arr)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────
class DOPAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE}  {APP_VERSION}")
        self.setMinimumSize(1280, 780)
        if os.path.exists(os.path.join(APP_DIR, "logo.ico")):
            self.setWindowIcon(QIcon(os.path.join(APP_DIR, "logo.ico")))

        # State
        self.mesh         = None
        self.mesh_path    = None
        self.cusp_points  = np.empty((0, 3))
        self.selected_pts = []
        self.monson_center = None
        self.plotter      = None
        self._worker      = None

        self._apply_palette()
        self._build_ui()
        self._update_button_states()

    # ── Palette ──────────────────────────────
    def _apply_palette(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {DARK_BG};
                color: {TEXT_PRIMARY};
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }}
            QGroupBox {{
                border: 1px solid {DARK_BORDER};
                border-radius: 8px;
                margin-top: 10px;
                font-size: 12px;
                font-weight: 700;
                color: {TEXT_MUTED};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QTableWidget {{
                background-color: {DARK_SURFACE};
                gridline-color: {DARK_BORDER};
                border: none;
                font-size: 12px;
                selection-background-color: {ACCENT_BLUE};
            }}
            QHeaderView::section {{
                background-color: {DARK_BORDER};
                color: {TEXT_PRIMARY};
                padding: 6px;
                border: none;
                font-weight: 700;
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 4px 8px;
            }}
            QStatusBar {{
                background-color: {DARK_SURFACE};
                color: {TEXT_MUTED};
                font-size: 12px;
                border-top: 1px solid {DARK_BORDER};
            }}
            QSplitter::handle {{
                background-color: {DARK_BORDER};
            }}
            QLabel#title_label {{
                font-size: 20px;
                font-weight: 700;
                color: {ACCENT_BLUE};
                padding: 4px 0;
            }}
            QLabel#subtitle_label {{
                font-size: 11px;
                color: {TEXT_MUTED};
            }}
            QFrame#divider {{
                background-color: {DARK_BORDER};
                max-height: 1px;
            }}
            QProgressBar {{
                border: 1px solid {DARK_BORDER};
                border-radius: 4px;
                background: {DARK_SURFACE};
                height: 6px;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT_BLUE};
                border-radius: 4px;
            }}
        """)

    # ── UI Layout ────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ──
        header = self._build_header()
        root_layout.addWidget(header)

        # ── Divider ──
        div = QFrame()
        div.setObjectName("divider")
        div.setFrameShape(QFrame.HLine)
        root_layout.addWidget(div)

        # ── Main splitter: sidebar | 3D view ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([280, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, stretch=1)

        # ── Status bar ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)   # indeterminate
        self.status_bar.addPermanentWidget(self.progress_bar)
        self._set_status("Ready — open a mesh file to begin.")

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"background-color: {DARK_SURFACE}; padding: 0 16px;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 8, 16, 8)

        title = QLabel(APP_TITLE)
        title.setObjectName("title_label")
        hl.addWidget(title)
        hl.addStretch()

        ver = QLabel(APP_VERSION)
        ver.setObjectName("subtitle_label")
        hl.addWidget(ver)
        return header

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(272)
        sidebar.setStyleSheet(f"background-color: {DARK_SURFACE};")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(12)

        # ── File Group ──
        file_grp = QGroupBox("  📂  File")
        fg_layout = QVBoxLayout(file_grp)
        fg_layout.setSpacing(8)

        self.btn_open = make_btn("Open Mesh File…", ACCENT_BLUE, "#fff", "#3a7af5", ACCENT_BLUE)
        self.btn_open.clicked.connect(self.open_file)
        fg_layout.addWidget(self.btn_open)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; padding: 4px 2px;")
        fg_layout.addWidget(self.lbl_file)
        layout.addWidget(file_grp)

        # ── Analysis Group ──
        ana_grp = QGroupBox("  🔬  Analysis")
        ag_layout = QVBoxLayout(ana_grp)
        ag_layout.setSpacing(8)

        self.btn_analyze = make_btn("Detect Cusps", "#50fa7b", "#0a1a0a", "#3ddd63", "#50fa7b")
        self.btn_analyze.clicked.connect(self.run_analysis)
        ag_layout.addWidget(self.btn_analyze)

        self.lbl_cusps = QLabel("Cusps detected: —")
        self.lbl_cusps.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; padding: 2px;")
        ag_layout.addWidget(self.lbl_cusps)
        layout.addWidget(ana_grp)

        # ── Monson Selection Group ──
        sel_grp = QGroupBox("  🎯  Monson Sphere — Pick 3 Cusps")
        sl_layout = QVBoxLayout(sel_grp)
        sl_layout.setSpacing(6)

        instr = QLabel(
            "Press <b>P</b> in the 3D view to pick:\n"
            " 1. Right molar DB cusp\n"
            " 2. Left molar DB cusp\n"
            " 3. Canine cusp"
        )
        instr.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; line-height: 1.5;")
        instr.setWordWrap(True)
        sl_layout.addWidget(instr)

        self.lbl_picked = QLabel("Picked: 0 / 3")
        self.lbl_picked.setStyleSheet(f"color: {ACCENT_AMBER}; font-size: 12px; font-weight: 700;")
        sl_layout.addWidget(self.lbl_picked)

        self.btn_reset_sel = make_btn("Reset Selection", "#44475a", TEXT_PRIMARY, "#5a5d72", DARK_BORDER)
        self.btn_reset_sel.clicked.connect(self.reset_selection)
        sl_layout.addWidget(self.btn_reset_sel)

        layout.addWidget(sel_grp)

        # ── Reset Group ──
        rst_grp = QGroupBox("  ↩  View")
        rl = QVBoxLayout(rst_grp)
        rl.setSpacing(8)

        self.btn_reset_view = make_btn("Reset Camera", "#44475a", TEXT_PRIMARY, "#5a5d72", DARK_BORDER)
        self.btn_reset_view.clicked.connect(self.reset_camera)
        rl.addWidget(self.btn_reset_view)

        self.btn_reset_all = make_btn("Clear All", ACCENT_RED, "#fff", "#dd3333", ACCENT_RED)
        self.btn_reset_all.clicked.connect(self.clear_all)
        rl.addWidget(self.btn_reset_all)
        layout.addWidget(rst_grp)

        layout.addStretch()

        # ── Footer ──
        footer = QLabel("DOPA  ·  Digital Occlusal Plane Analyzer\nFor clinical decision support only.")
        footer.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        return sidebar

    def _build_right_panel(self):
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {DARK_BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── PyVista 3D viewport ──
        self.plotter = BackgroundPlotter(
            show=False,
            app=QApplication.instance(),
        )
        self.plotter.set_background("#0d0d1a")
        self.plotter.enable_anti_aliasing()
        self.plotter.add_axes(interactive=True)

        vp_widget = self.plotter.app_window
        vp_widget.setParent(panel)
        layout.addWidget(vp_widget, stretch=3)

        # ── Divider ──
        div = QFrame()
        div.setObjectName("divider")
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"background-color: {DARK_BORDER}; max-height: 1px;")
        layout.addWidget(div)

        # ── Results table ──
        tbl_header = QLabel("  📊  Cusp Deviation Report")
        tbl_header.setStyleSheet(
            f"background-color: {DARK_SURFACE}; color: {TEXT_PRIMARY}; "
            f"font-size: 13px; font-weight: 700; padding: 8px 12px;"
        )
        layout.addWidget(tbl_header)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Cusp #", "X (mm)", "Y (mm)", "Deviation (mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            self.table.styleSheet() +
            f"QTableWidget {{ alternate-background-color: {DARK_BG}; }}"
        )
        self.table.setMaximumHeight(200)
        layout.addWidget(self.table, stretch=0)

        return panel

    # ── Helpers ──────────────────────────────
    def _set_status(self, msg, color=None):
        color = color or TEXT_MUTED
        self.status_bar.setStyleSheet(f"color: {color};")
        self.status_bar.showMessage(f"  {msg}")

    def _update_button_states(self):
        has_mesh  = self.mesh is not None
        has_cusps = len(self.cusp_points) > 0
        self.btn_analyze.setEnabled(has_mesh)
        self.btn_reset_sel.setEnabled(has_cusps)
        self.btn_reset_view.setEnabled(has_mesh)
        self.btn_reset_all.setEnabled(has_mesh)

    # ── Actions ──────────────────────────────
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Mesh File",
            APP_DIR,
            "3D Mesh files (*.obj *.stl *.ply *.vtk *.vtp);;All files (*.*)"
        )
        if not path:
            return
        self._load_mesh(path)

    def _load_mesh(self, path):
        self._set_status(f"Loading {os.path.basename(path)}…")
        self.progress_bar.setVisible(True)
        QApplication.processEvents()
        try:
            self.mesh = pv.read(path)
            self.mesh_path = path
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load file:\n{e}")
            self.progress_bar.setVisible(False)
            return

        # Reset state
        self.cusp_points   = np.empty((0, 3))
        self.selected_pts  = []
        self.monson_center = None
        self.table.setRowCount(0)
        self.lbl_picked.setText("Picked: 0 / 3")

        # Display mesh
        self.plotter.clear()
        self.plotter.add_mesh(
            self.mesh,
            color="#c8c8d8",
            specular=0.5,
            smooth_shading=True,
            opacity=1.0,
            name="main_mesh"
        )
        self.plotter.reset_camera()

        short = os.path.basename(path)
        pts   = self.mesh.n_points
        self.lbl_file.setText(f"{short}\n{pts:,} points")
        self.lbl_cusps.setText("Cusps detected: —")
        self._set_status(f"Loaded: {short}  ({pts:,} points)", ACCENT_GREEN)
        self.progress_bar.setVisible(False)
        self._update_button_states()

    def run_analysis(self):
        if self.mesh is None:
            return
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._set_status("Detecting cusp tips…")

        self._worker = CuspDetector(self.mesh.points.copy())
        self._worker.progress.connect(lambda m: self._set_status(m))
        self._worker.finished.connect(self._on_cusps_detected)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_cusps_detected(self, cusps):
        self.cusp_points = cusps
        self.progress_bar.setVisible(False)

        # Remove old cusp actors then re-add
        self.plotter.remove_actor("cusps")
        if len(cusps) > 0:
            cloud = pv.PolyData(cusps)
            self.plotter.add_mesh(
                cloud,
                render_points_as_spheres=True,
                point_size=10,
                color=ACCENT_GREEN,
                name="cusps"
            )

        n = len(cusps)
        self.lbl_cusps.setText(f"Cusps detected: {n}")
        self._set_status(f"Cusp detection complete — {n} tips found.", ACCENT_GREEN)
        self._update_button_states()

        # Re-enable picking
        self.plotter.enable_point_picking(
            callback=self._pick_callback,
            show_message=False,
            show_point=True,
            color="yellow",
            point_size=14,
        )
        self.btn_analyze.setEnabled(True)

    def _on_analysis_error(self, msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", msg)
        self.btn_analyze.setEnabled(True)
        self._set_status("Analysis failed.", ACCENT_RED)

    def _pick_callback(self, point):
        if point is None:
            return
        if len(self.selected_pts) >= 3:
            self._set_status("Already have 3 points. Reset selection to pick again.", ACCENT_AMBER)
            return

        self.selected_pts.append(np.array(point))
        n = len(self.selected_pts)
        self.lbl_picked.setText(f"Picked: {n} / 3")
        self._set_status(f"Point {n} selected at ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")

        if n == 3:
            self._compute_monson()

    def _compute_monson(self):
        self._set_status("Computing Monson sphere…")
        QApplication.processEvents()

        pts = np.array(self.selected_pts)
        R   = MONSON_RADIUS

        def equations(center):
            return [np.linalg.norm(pts[i] - center) - R for i in range(3)]

        center_guess = np.mean(pts, axis=0)
        center = fsolve(equations, center_guess)
        self.monson_center = center

        # Draw sphere
        self.plotter.remove_actor("monson_sphere")
        sphere = pv.Sphere(radius=R, center=center, theta_resolution=80, phi_resolution=80)
        self.plotter.add_mesh(sphere, color="#5588ff", opacity=0.25, name="monson_sphere")

        # Compute & display deviations
        self._populate_table(center)
        self._set_status(
            f"Monson sphere centre: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})  "
            f"| Radius: {R} mm",
            ACCENT_BLUE
        )
        self.plotter.render()

    def _populate_table(self, center):
        self.table.setRowCount(0)
        self.plotter.remove_actor("dev_markers")

        if len(self.cusp_points) == 0:
            return

        cloud_pts   = []
        cloud_colors = []

        for i, cusp in enumerate(self.cusp_points):
            dev    = np.linalg.norm(cusp - center) - MONSON_RADIUS
            status = "above" if dev > 0 else "below"
            color  = ACCENT_RED if dev > 0 else ACCENT_BLUE

            row = self.table.rowCount()
            self.table.insertRow(row)

            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(f"{cusp[0]:.2f}"),
                QTableWidgetItem(f"{cusp[1]:.2f}"),
                QTableWidgetItem(f"{dev:+.2f} mm  ({status})"),
            ]
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                if col == 3:
                    item.setForeground(QColor(color))
                self.table.setItem(row, col, item)

            cloud_pts.append(cusp)
            cloud_colors.append([255, 80, 80, 255] if dev > 0 else [80, 130, 255, 255])

        # Render deviation markers
        if cloud_pts:
            cloud = pv.PolyData(np.array(cloud_pts))
            rgba  = np.array(cloud_colors, dtype=np.uint8)
            cloud["colors"] = rgba
            self.plotter.add_mesh(
                cloud,
                render_points_as_spheres=True,
                point_size=12,
                scalars="colors",
                rgb=True,
                name="dev_markers"
            )

    def reset_selection(self):
        self.selected_pts  = []
        self.monson_center = None
        self.lbl_picked.setText("Picked: 0 / 3")
        self.table.setRowCount(0)
        self.plotter.remove_actor("monson_sphere")
        self.plotter.remove_actor("dev_markers")
        # Re-show original green cusp markers
        self.plotter.remove_actor("cusps")
        if len(self.cusp_points) > 0:
            cloud = pv.PolyData(self.cusp_points)
            self.plotter.add_mesh(
                cloud, render_points_as_spheres=True,
                point_size=10, color=ACCENT_GREEN, name="cusps"
            )
        self._set_status("Selection reset — press P in the 3D view to pick cusps.")

    def reset_camera(self):
        if self.plotter:
            self.plotter.reset_camera()

    def clear_all(self):
        reply = QMessageBox.question(
            self, "Clear All",
            "Clear the loaded mesh and all analysis results?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.mesh          = None
        self.mesh_path     = None
        self.cusp_points   = np.empty((0, 3))
        self.selected_pts  = []
        self.monson_center = None
        self.plotter.clear()
        self.table.setRowCount(0)
        self.lbl_file.setText("No file loaded")
        self.lbl_cusps.setText("Cusps detected: —")
        self.lbl_picked.setText("Picked: 0 / 3")
        self._set_status("Cleared — open a mesh file to begin.")
        self._update_button_states()

    def closeEvent(self, event):
        if self.plotter:
            self.plotter.close()
        event.accept()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DOPA")
    app.setApplicationVersion(APP_VERSION)

    window = DOPAWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
