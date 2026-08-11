"""
DOPA - Desktop Occlusal Plane Analyzer
Rewritten with:
  - PyQt5 main window visible immediately on startup
  - Background mesh loading via QThread (no UI freeze)
  - KDTree-based O(n log n) cusp detection (replaces O(n²) loop)
  - Animated spinner during processing
"""

import sys
import os
import threading

import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import pyvista as pv
from pyvistaqt import BackgroundPlotter   # embeds VTK inside Qt window

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar, QProgressBar,
    QSizePolicy, QFrame, QSplitter, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)
from PyQt5.QtGui import QFont, QIcon, QMovie, QPixmap

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MONSON_RADIUS   = 101.6   # mm
CUSP_RADIUS     = 2.0     # local neighbourhood radius for local-max search
DBSCAN_EPS      = 3.0     # mm  – merge nearby cusp candidates
DBSCAN_MIN_PTS  = 1

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MESH = os.path.join(APP_DIR, "mandible.obj.obj")


# ---------------------------------------------------------------------------
# CUSP DETECTION  –  O(n log n) via cKDTree + DBSCAN cluster pruning
# ---------------------------------------------------------------------------
def detect_cusps_kdtree(points: np.ndarray,
                         radius: float = CUSP_RADIUS,
                         z_percentile: float = 85,
                         dbscan_eps: float = DBSCAN_EPS) -> np.ndarray:
    """
    1. Keep only the top z_percentile of points  (occlusal surface)
    2. Build a cKDTree over those points
    3. For each point, query neighbours within `radius` – O(n log n) total
    4. A point is a local maximum if its Z is the max in its neighbourhood
    5. DBSCAN merges nearby local maxima → one representative per cusp
    Returns array of shape (N, 3).
    """
    z_vals = points[:, 2]
    z_thresh = np.percentile(z_vals, z_percentile)
    occ = points[z_vals > z_thresh]

    if len(occ) == 0:
        return np.empty((0, 3))

    tree = cKDTree(occ)

    # Query all neighbours at once – returns list of index arrays
    neighbour_lists = tree.query_ball_point(occ, r=radius)

    # Vectorised local-max check
    local_max_mask = np.array([
        occ[i, 2] >= np.max(occ[idxs, 2])
        for i, idxs in enumerate(neighbour_lists)
    ], dtype=bool)

    candidates = occ[local_max_mask]

    if len(candidates) == 0:
        return candidates

    # DBSCAN: merge candidates that are within dbscan_eps of each other
    labels = DBSCAN(eps=dbscan_eps, min_samples=DBSCAN_MIN_PTS,
                    algorithm='ball_tree').fit_predict(candidates)

    cusps = []
    for lbl in set(labels):
        cluster = candidates[labels == lbl]
        # Pick the highest point in each cluster as the cusp tip
        cusps.append(cluster[np.argmax(cluster[:, 2])])

    return np.array(cusps)


# ---------------------------------------------------------------------------
# MONSON SPHERE SOLVER
# ---------------------------------------------------------------------------
def compute_monson(three_points: np.ndarray) -> np.ndarray:
    p1, p2, p3 = three_points
    R = MONSON_RADIUS

    def equations(center):
        return [
            np.linalg.norm(p1 - center) - R,
            np.linalg.norm(p2 - center) - R,
            np.linalg.norm(p3 - center) - R,
        ]

    center_guess = np.mean(three_points, axis=0)
    return np.array(fsolve(equations, center_guess))


# ---------------------------------------------------------------------------
# BACKGROUND WORKER THREAD
# ---------------------------------------------------------------------------
class MeshLoader(QThread):
    """Loads mesh and detects cusps in a background thread."""
    progress_msg  = pyqtSignal(str)          # status text updates
    finished_ok   = pyqtSignal(object, object)  # (mesh, cusp_points)
    finished_err  = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.progress_msg.emit("Loading mesh from disk …")
            mesh = pv.read(self.path)
            points = mesh.points

            self.progress_msg.emit(
                f"Mesh loaded — {len(points):,} vertices. Detecting cusps …"
            )

            cusps = detect_cusps_kdtree(points)

            self.progress_msg.emit(
                f"Done — {len(cusps)} cusp tip(s) detected."
            )
            self.finished_ok.emit(mesh, cusps)

        except Exception as exc:
            self.finished_err.emit(str(exc))


# ---------------------------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------------------------
class DOPAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SanVie")
        self.resize(1280, 800)
        self._set_icon()

        self.mesh       = None
        self.cusp_pts   = None
        self.selected   = []
        self.loader     = None
        self._spinner_angle = 0

        self._build_ui()
        self._start_spinner()

    # ------------------------------------------------------------------
    # Icon
    # ------------------------------------------------------------------
    def _set_icon(self):
        ico = os.path.join(APP_DIR, "logo.ico")
        if os.path.exists(ico):
            self.setWindowIcon(QIcon(ico))

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Top toolbar ────────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setFixedHeight(56)
        toolbar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                border-bottom: 1px solid #0f3460;
            }
        """)
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(14, 0, 14, 0)

        title_lbl = QLabel("Sanvie 🦷  3D Digital Occlusion Plane Analyser")
        title_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_lbl.setStyleSheet("color: #e2e8f0; background: transparent;")
        tb_lay.addWidget(title_lbl)

        tb_lay.addStretch()

        self.open_btn = QPushButton("📂  Open Mesh")
        self.open_btn.setFixedHeight(36)
        self.open_btn.setCursor(Qt.PointingHandCursor)
        self.open_btn.setStyleSheet("""
            QPushButton {
                background: #0f3460;
                color: #e2e8f0;
                border: 1px solid #e94560;
                border-radius: 6px;
                padding: 0 18px;
                font-size: 13px;
            }
            QPushButton:hover  { background: #e94560; }
            QPushButton:pressed{ background: #c73652; }
            QPushButton:disabled{ background: #2d3748; color: #718096; border-color: #4a5568; }
        """)
        self.open_btn.clicked.connect(self._open_file_dialog)
        tb_lay.addWidget(self.open_btn)

        self.reset_btn = QPushButton("🔄  Reset")
        self.reset_btn.setFixedHeight(36)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet(self.open_btn.styleSheet())
        self.reset_btn.clicked.connect(self._reset)
        tb_lay.addWidget(self.reset_btn)

        root_layout.addWidget(toolbar)

        # ── View presets bar ───────────────────────────────────────────
        view_bar = QFrame()
        view_bar.setFixedHeight(40)
        view_bar.setStyleSheet("""
            QFrame {
                background: #16213e;
                border-bottom: 1px solid #0f3460;
            }
        """)
        vb_lay = QHBoxLayout(view_bar)
        vb_lay.setContentsMargins(10, 0, 10, 0)
        vb_lay.setSpacing(6)

        view_lbl = QLabel("View:")
        view_lbl.setStyleSheet("color: #718096; font-size: 11px; background: transparent;")
        vb_lay.addWidget(view_lbl)

        view_btn_style = """
            QPushButton {
                background: #1a1a2e;
                color: #a0aec0;
                border: 1px solid #2d3748;
                border-radius: 4px;
                padding: 0 12px;
                font-size: 11px;
                min-width: 56px;
            }
            QPushButton:hover   { background: #0f3460; color: #e2e8f0; border-color: #4299e1; }
            QPushButton:pressed { background: #063070; }
            QPushButton:disabled{ color: #4a5568; border-color: #1a202c; }
        """

        self._view_buttons = []
        for label, view in [
            ("⬆ Front",  "front"),
            ("⬇ Back",   "back"),
            ("▲ Top",    "top"),
            ("▼ Bottom", "bottom"),
            ("◀ Left",   "left"),
            ("▶ Right",  "right"),
            ("⤢ ISO",    "iso"),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.setEnabled(False)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(view_btn_style)
            btn.clicked.connect(lambda checked, v=view: self._set_view(v))
            vb_lay.addWidget(btn)
            self._view_buttons.append(btn)

        vb_lay.addStretch()
        root_layout.addWidget(view_bar)

        # ── Splitter: VTK view  |  Info panel ─────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #0f3460; }")

        # VTK / overlay area
        self.vtk_container = QWidget()
        self.vtk_container.setMinimumWidth(600)
        self.vtk_container.setStyleSheet("background: #0d1117;")
        vtk_lay = QVBoxLayout(self.vtk_container)
        vtk_lay.setContentsMargins(0, 0, 0, 0)

        # Placeholder shown before mesh is loaded
        self.placeholder = QWidget()
        self.placeholder.setStyleSheet("background: #0d1117;")
        ph_lay = QVBoxLayout(self.placeholder)
        ph_lay.setAlignment(Qt.AlignCenter)
        ph_lay.setSpacing(24)

        ph_icon = QLabel("🦷")
        ph_icon.setAlignment(Qt.AlignCenter)
        ph_icon.setFont(QFont("Segoe UI", 52))
        ph_icon.setStyleSheet("background: transparent; color: #2d3748;")
        ph_lay.addWidget(ph_icon)

        ph_text = QLabel(
            "No mesh loaded"
        )
        ph_text.setAlignment(Qt.AlignCenter)
        ph_text.setFont(QFont("Segoe UI", 16))
        ph_text.setStyleSheet("color: #4a5568; background: transparent;")
        ph_lay.addWidget(ph_text)

        ph_sub = QLabel("Supported formats: OBJ · STL · PLY · VTK · VTP")
        ph_sub.setAlignment(Qt.AlignCenter)
        ph_sub.setFont(QFont("Segoe UI", 10))
        ph_sub.setStyleSheet("color: #2d3748; background: transparent;")
        ph_lay.addWidget(ph_sub)

        self.ph_open_btn = QPushButton("📂   Open Mesh File")
        self.ph_open_btn.setFixedSize(220, 48)
        self.ph_open_btn.setCursor(Qt.PointingHandCursor)
        self.ph_open_btn.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.ph_open_btn.setStyleSheet("""
            QPushButton {
                background: #e94560;
                color: #ffffff;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover  { background: #c73652; }
            QPushButton:pressed{ background: #a02840; }
        """)
        self.ph_open_btn.clicked.connect(self._open_file_dialog)
        ph_lay.addWidget(self.ph_open_btn, alignment=Qt.AlignCenter)

        vtk_lay.addWidget(self.placeholder)

        # Processing overlay (hidden initially)
        self.processing_overlay = QWidget(self.vtk_container)
        self.processing_overlay.setStyleSheet(
            "background: rgba(13,17,23,210); border-radius: 12px;"
        )
        overlay_lay = QVBoxLayout(self.processing_overlay)
        overlay_lay.setAlignment(Qt.AlignCenter)

        self.spinner_lbl = QLabel("⏳")
        self.spinner_lbl.setAlignment(Qt.AlignCenter)
        self.spinner_lbl.setFont(QFont("Segoe UI", 48))
        overlay_lay.addWidget(self.spinner_lbl)

        self.proc_lbl = QLabel("Initialising …")
        self.proc_lbl.setAlignment(Qt.AlignCenter)
        self.proc_lbl.setFont(QFont("Segoe UI", 13))
        self.proc_lbl.setStyleSheet("color: #e2e8f0; background: transparent;")
        overlay_lay.addWidget(self.proc_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)          # indeterminate
        self.progress_bar.setFixedWidth(320)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #1a202c;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #0f3460);
            }
        """)
        overlay_lay.addWidget(self.progress_bar, alignment=Qt.AlignCenter)
        self.processing_overlay.hide()

        splitter.addWidget(self.vtk_container)

        # ── Info/results panel ──────────────────────────────────────────
        info_panel = QFrame()
        info_panel.setFixedWidth(280)
        info_panel.setStyleSheet("""
            QFrame {
                background: #111827;
                border-left: 1px solid #0f3460;
            }
        """)
        info_lay = QVBoxLayout(info_panel)
        info_lay.setContentsMargins(16, 16, 16, 16)
        info_lay.setSpacing(12)

        section_style = (
            "color: #e94560; font-size: 11px; font-weight: bold; "
            "text-transform: uppercase; letter-spacing: 1px; "
            "background: transparent;"
        )
        val_style = "color: #e2e8f0; font-size: 13px; background: transparent;"

        # ── Cusp count ──────────────────────────────
        info_lay.addWidget(self._section("Detected Cusps", section_style))
        self.cusp_count_lbl = QLabel("—")
        self.cusp_count_lbl.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.cusp_count_lbl.setStyleSheet("color: #68d391; background: transparent;")
        info_lay.addWidget(self.cusp_count_lbl)

        info_lay.addWidget(self._section("Monson Sphere", section_style))

        self.monson_lbl = QLabel(
            "Select 3 cusp tips in the 3-D view\n"
            "(hover + press P) to compute the\nMonson sphere."
        )
        self.monson_lbl.setWordWrap(True)
        self.monson_lbl.setStyleSheet(val_style)
        self.monson_lbl.setFont(QFont("Segoe UI", 11))
        info_lay.addWidget(self.monson_lbl)

        info_lay.addWidget(self._section("Selection", section_style))
        self.sel_lbl = QLabel("0 / 3 cusps selected")
        self.sel_lbl.setStyleSheet(val_style)
        self.sel_lbl.setFont(QFont("Segoe UI", 11))
        info_lay.addWidget(self.sel_lbl)

        info_lay.addWidget(self._section("Deviation Report", section_style))

        dev_scroll = QScrollArea()
        dev_scroll.setWidgetResizable(True)
        dev_scroll.setFrameShape(QFrame.NoFrame)
        dev_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #1a202c;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #4a5568;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #e94560;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.dev_lbl = QLabel("—")
        self.dev_lbl.setWordWrap(True)
        self.dev_lbl.setStyleSheet(val_style)
        self.dev_lbl.setFont(QFont("Segoe UI", 10))
        self.dev_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        dev_scroll.setWidget(self.dev_lbl)
        info_lay.addWidget(dev_scroll, 1)

        # Instructions
        hint = QLabel(
            "Legend\n"
            "🟢 Green  = auto-detected cusp\n"
            "🟡 Yellow = selected by you\n"
            "🔵 Blue   = below Monson sphere\n"
            "🔴 Red    = above Monson sphere\n\n"
            "Mouse controls\n"
            "🖱 Left-drag   = rotate\n"
            "🖱 Scroll      = zoom\n"
            "🖱 Shift+drag   = pan\n"
            "⌨ Right click/ P key = pick cusp under cursor"
        )
        hint.setWordWrap(True)
        hint.setFont(QFont("Segoe UI", 10))
        hint.setStyleSheet("color: #718096; background: transparent;")
        info_lay.addWidget(hint)

        splitter.addWidget(info_panel)
        root_layout.addWidget(splitter, 1)

        # ── Status bar ─────────────────────────────────────────────────
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #1a1a2e;
                color: #a0aec0;
                font-size: 11px;
                border-top: 1px solid #0f3460;
            }
        """)
        self.status_bar.showMessage("Ready — open a mesh file to begin.")

        # pyvistaqt plotter (created lazily when mesh is loaded)
        self.plotter = None

    def _section(self, text, style) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        return lbl

    # ------------------------------------------------------------------
    # Spinner animation (rotates emoji via timer)
    # ------------------------------------------------------------------
    def _start_spinner(self):
        self._spinner_frames = ["⏳", "⌛", "⏳", "⌛"]
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_timer.setInterval(400)

    def _tick_spinner(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self.spinner_lbl.setText(self._spinner_frames[self._spinner_idx])

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------
    def _show_overlay(self, msg: str = "Processing …"):
        self.proc_lbl.setText(msg)
        # Position overlay centred over the vtk container
        self.processing_overlay.setGeometry(self.vtk_container.rect())
        self.processing_overlay.show()
        self.processing_overlay.raise_()
        self._spinner_timer.start()

    def _hide_overlay(self):
        self._spinner_timer.stop()
        self.processing_overlay.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.processing_overlay.isVisible():
            self.processing_overlay.setGeometry(self.vtk_container.rect())

    # ------------------------------------------------------------------
    # File open
    # ------------------------------------------------------------------
    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Mesh File",
            APP_DIR,
            "3D Mesh files (*.obj *.stl *.ply *.vtk *.vtp);;"
            "OBJ files (*.obj);;"
            "STL files (*.stl);;"
            "All files (*.*)"
        )
        if not path:
            return
        self._load_mesh(path)

    # ------------------------------------------------------------------
    # Mesh loading pipeline
    # ------------------------------------------------------------------
    def _load_mesh(self, path: str):
        self.open_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        for b in self._view_buttons:
            b.setEnabled(False)
        self.status_bar.showMessage(f"Loading: {os.path.basename(path)} …")
        self._show_overlay("Reading mesh and detecting cusp tips …")

        self.loader = MeshLoader(path, parent=self)
        self.loader.progress_msg.connect(self._on_loader_progress)
        self.loader.finished_ok.connect(self._on_load_done)
        self.loader.finished_err.connect(self._on_load_error)
        self.loader.start()

    def _on_loader_progress(self, msg: str):
        self.proc_lbl.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_load_done(self, mesh, cusp_pts):
        self._hide_overlay()
        self.mesh     = mesh
        self.cusp_pts = cusp_pts
        self.selected = []

        self.cusp_count_lbl.setText(str(len(cusp_pts)))
        self.sel_lbl.setText("0 / 3 cusps selected")
        self.dev_lbl.setText("—")
        self.monson_lbl.setText(
            "Select 3 cusp tips in the 3-D view\n(hover + press P)."
        )

        self.status_bar.showMessage(
            f"Loaded — {len(mesh.points):,} vertices, "
            f"{len(cusp_pts)} cusp tip(s) detected."
        )

        self.open_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        for b in self._view_buttons:
            b.setEnabled(True)

        self._build_plotter()

    def _on_load_error(self, err: str):
        self._hide_overlay()
        self.status_bar.showMessage(f"Error: {err}")
        self.open_btn.setEnabled(True)
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Load Error", f"Could not load mesh:\n{err}")

    # ------------------------------------------------------------------
    # Camera view presets
    # ------------------------------------------------------------------
    def _set_view(self, view: str):
        if self.plotter is None:
            return
        presets = {
            "front":  dict(position=(0, -1, 0),  viewup=(0, 0, 1)),
            "back":   dict(position=(0,  1, 0),  viewup=(0, 0, 1)),
            "top":    dict(position=(0,  0, 1),  viewup=(0, 1, 0)),
            "bottom": dict(position=(0,  0, -1), viewup=(0, 1, 0)),
            "left":   dict(position=(-1, 0, 0),  viewup=(0, 0, 1)),
            "right":  dict(position=( 1, 0, 0),  viewup=(0, 0, 1)),
            "iso":    dict(position=(1, -1, 1),   viewup=(0, 0, 1)),
        }
        if view not in presets:
            return
        p = presets[view]
        # Scale direction to fit the current model bounds
        bounds = self.mesh.bounds          # (xmin,xmax,ymin,ymax,zmin,zmax)
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        diag = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ) * 2.5
        dx, dy, dz = p["position"]
        self.plotter.camera.focal_point = (cx, cy, cz)
        self.plotter.camera.position    = (cx + dx * diag,
                                           cy + dy * diag,
                                           cz + dz * diag)
        self.plotter.camera.up          = p["viewup"]
        self.plotter.reset_camera()
        self.plotter.render()

    # ------------------------------------------------------------------
    # Build / refresh the VTK plotter
    # ------------------------------------------------------------------
    def _build_plotter(self):
        # Remove old plotter if exists
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

        # Remove placeholder
        self.placeholder.hide()

        # Create embedded plotter
        self.plotter = BackgroundPlotter(show=False)
        self.plotter.set_background("#0d1117")

        # Add mesh
        self.plotter.add_mesh(self.mesh, color="#b0bec5", opacity=0.92,
                               smooth_shading=True)

        # Add green cusp markers
        for c in self.cusp_pts:
            self.plotter.add_mesh(
                pv.Sphere(radius=0.8, center=c), color="#68d391"
            )

        # Enable trackball camera style (supports scroll wheel zoom & pan)
        self.plotter.enable_trackball_style()

        # Enable picking — left_clicking=False keeps mouse buttons free
        # for camera navigation (rotate / zoom / pan).
        # Users pick a cusp by hovering over it and pressing the P key.
        try:
            self.plotter.enable_point_picking(
                callback=self._pick_callback,
                show_message=False,
                show_point=True,
                tolerance=0.025,
                left_clicking=False,   # ← do NOT hijack mouse events
            )
        except TypeError:
            # Older PyVista versions that don't accept left_clicking
            self.plotter.enable_point_picking(
                callback=self._pick_callback,
                show_message=False,
                show_point=True,
                tolerance=0.025,
            )

        # Embed Qt widget
        vtk_lay = self.vtk_container.layout()
        vtk_lay.addWidget(self.plotter)

    # ------------------------------------------------------------------
    # Point picking
    # ------------------------------------------------------------------
    def _pick_callback(self, point):
        if point is None:
            return
        self.selected.append(np.array(point))
        n = len(self.selected)
        self.sel_lbl.setText(f"{n} / 3 cusps selected")

        self.plotter.add_mesh(
            pv.Sphere(radius=1.2, center=point), color="#f6e05e"
        )

        if n == 3:
            self._compute_and_render_monson()

    def _compute_and_render_monson(self):
        pts    = np.array(self.selected)
        center = compute_monson(pts)

        # Draw Monson sphere
        # ambient=1.0 / diffuse=0.0: render in true flat colour so VTK
        # lighting doesn't darken the sphere against the dark background.
        sphere = pv.Sphere(radius=MONSON_RADIUS, center=center,
                           theta_resolution=120, phi_resolution=120)
        self.plotter.add_mesh(sphere, color="#7cced9", opacity=0.10,
                              ambient=1.0, diffuse=0.0, specular=0.0)

        # Deviation analysis
        lines = []
        for cusp in self.cusp_pts:
            deviation = np.linalg.norm(cusp - center) - MONSON_RADIUS
            is_above = deviation > 0
            color = "#fc8181" if is_above else "#63b3ed"
            badge_color = "#dc2626" if is_above else "#2563eb"
            status = "above" if is_above else "below"

            self.plotter.add_mesh(
                pv.Sphere(radius=0.9, center=cusp), color=color
            )
            self.plotter.add_point_labels(
                [cusp],
                [f"{deviation:+.2f} mm"],
                point_size=16,
                text_color="#ffffff",
                shape_color=badge_color,
                shape_opacity=0.9,
                font_size=11,
                bold=True,
            )
            lines.append(f"  {status}  {deviation:+.2f} mm")

        self.monson_lbl.setText(
            f"Centre: {center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}\n"
            f"Radius: {MONSON_RADIUS} mm"
        )
        self.dev_lbl.setText("\n".join(lines) if lines else "No data")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset(self):
        if self.mesh is None:
            return
        self.selected = []
        self.sel_lbl.setText("0 / 3 cusps selected")
        self.dev_lbl.setText("—")
        self.monson_lbl.setText(
            "Select 3 cusp tips in the 3-D view\n(hover + press P)."
        )
        # Rebuild plotter from scratch (clean slate)
        self._build_plotter()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("DOPA")
    app.setStyle("Fusion")

    win = DOPAWindow()
    win.show()                      # ← window is immediately visible


    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
