# 🦷 DOPA – Desktop Occlusal Plane Analyzer

**DOPA (Desktop Occlusal Plane Analyzer)** is a desktop application designed for dental professionals and researchers to load 3D dental mesh scans (`.stl`, `.obj`, `.ply`, `.vtk`, `.vtp`), automatically detect occlusal cusp tips using KDTree spatial indexing and DBSCAN clustering, and calculate the **Monson Sphere** (101.6 mm radius) plane deviation report.

---

## 🌟 Key Features

* **Non-Blocking Instant UI**: Powered by PyQt5 with background worker threads (`QThread`), ensuring smooth startup and UI responsiveness during heavy 3D mesh processing.
* **Fast Cusp Detection**: Employs an $O(n \log n)$ KDTree spatial search combined with DBSCAN clustering to identify distinct occlusal cusp tips efficiently.
* **Interactive 3D Visualization**: Built on PyVista Qt (`BackgroundPlotter`) for interactive panning, zooming, rotating, and point-picking.
* **Monson Sphere Fitting**: Allows users to select 3 cusp tips (via hover + keypress `P`) to compute the 101.6 mm Monson sphere center and individual cusp deviation distances.
* **Color-Coded Analysis**:
  * 🟢 **Green Spheres**: Auto-detected cusp tips.
  * 🟡 **Yellow Spheres**: User-selected reference cusps.
  * 🔵 **Blue Spheres**: Cusps located below the Monson sphere.
  * 🔴 **Red Spheres**: Cusps located above the Monson sphere.
* **Camera View Presets**: One-click camera alignment for Front, Back, Top, Bottom, Left, Right, and Isometric views.
* **Standalone Executable**: Easy build process using PyInstaller to package into a single `.exe` for distribution to non-technical users.

---

## 📁 Repository Structure

```
DOPAPS/
├── stl_viewer.py            # Main application source code (PyQt5 GUI & PyVista viewer)
├── logo.ico                 # Application icon file
├── mandible.obj.obj         # Sample 3D mandibular mesh model
├── DOPA.spec                # PyInstaller specification file for bundling .exe
├── requirements.txt         # Project Python dependencies
├── README.md                # Project documentation
├── SETUP_AND_BUILD_GUIDE.md # Beginner-friendly step-by-step setup & executable creation guide
└── .gitignore               # Excluded build artifacts, environments, and agent files
```

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your Windows system.

### 2. Setup & Installation
1. Clone or download this repository.
2. Open Command Prompt or PowerShell in the project directory.
3. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### 3. Run the Application
```powershell
python stl_viewer.py
```

---

## 🛠️ Building Standalone `.exe` for Distribution

To package DOPA into a standalone Windows executable (`DOPA.exe`):

1. Activate your virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
2. Run PyInstaller:
   ```powershell
   pyinstaller --noconfirm --onefile --windowed --icon=logo.ico --name=DOPA stl_viewer.py
   ```
3. Find your standalone folder inside `dist/DOPA/`. Double-click `DOPA.exe` to run.

> 📖 **Need detailed step-by-step instructions for non-technical users?**  
> Check out the complete [SETUP_AND_BUILD_GUIDE.md](file:///d:/GST%20apps/DOPAPS/DOPAPS/SETUP_AND_BUILD_GUIDE.md) included in this repository.

---

## 📖 How to Use DOPA

1. **Load a 3D Mesh**: Click **📂 Open Mesh** or use the center button to select a `.stl`, `.obj`, or `.ply` file.
2. **View Detected Cusps**: The application automatically processes the mesh and highlights detected cusp tips with 🟢 **Green** spheres.
3. **Select 3 Cusp Tips**:
   - Hover your mouse cursor over a green cusp tip in the 3D viewport.
   - Press **`P`** on your keyboard to select that point. Selected points turn 🟡 **Yellow**.
   - Repeat until 3 points are selected.
4. **View Analysis Report**:
   - Upon selecting the 3rd point, DOPA automatically computes the **Monson Sphere** (101.6 mm radius).
   - A deviation report appears in the right-hand side panel showing exact deviations in millimeters (+ for above, - for below).
5. **Reset**: Click **🔄 Reset** to clear selections and restart analysis.

---

## 📄 License & Notes
- Developed for Dental Occlusal Analysis.
- Proprietary / Open Source (as designated by author).
