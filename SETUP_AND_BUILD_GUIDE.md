# 📘 Complete Setup, Local Run & Executable Distribution Guide

Welcome! This document provides **step-by-step instructions** to set up **DOPA (Desktop Occlusal Plane Analyzer)** on Windows, run it locally on your computer, build a standalone `.exe` installer/application, and distribute it to non-technical users (such as dentists or clinical staff) so they can run it with a simple double-click without installing Python.

---

## 📋 Table of Contents
1. [Prerequisites (One-Time Setup)](#1-prerequisites-one-time-setup)
2. [Step 1: Download Code from GitHub](#step-1-download-code-from-github)
3. [Step 2: Set Up Python Environment & Install Dependencies](#step-2-set-up-python-environment--install-dependencies)
4. [Step 3: Run the Application Locally](#step-3-run-the-application-locally)
5. [Step 4: Build a Standalone Executable (.exe)](#step-4-build-a-standalone-executable-exe)
6. [Step 5: Distribute to Non-Technical End Users](#step-5-distribute-to-non-technical-end-users)
7. [❓ Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 1. Prerequisites (One-Time Setup)

Before starting, ensure your Windows computer has **Python** installed.

### Installing Python:
1. Go to the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download **Python 3.10** or higher (e.g., Python 3.11 or 3.12).
3. Run the installer `.exe` file.
4. ⚠️ **CRITICAL STEP**: On the very first screen of the installer, check the box at the bottom that says **"Add python.exe to PATH"**.
   > *If you do not check this box, Windows will not recognize `python` commands in Command Prompt.*
5. Click **Install Now** and wait for it to complete.

---

## Step 1: Download Code from GitHub

You can get the code using either of the two methods below:

### Method A: Download as ZIP File (Easiest for Non-Tech Users)
1. Open the GitHub repository page in your browser.
2. Click the green **`Code`** button near the top right.
3. Select **`Download ZIP`**.
4. Once downloaded, right-click the ZIP file → select **Extract All...** → click **Extract**.
5. Remember the folder location where you extracted the files (e.g., `C:\DOPAPS`).

### Method B: Using Git Clone (For Developers)
Open Command Prompt (`cmd`) or PowerShell and run:
```powershell
git clone https://github.com/YOUR_USERNAME/DOPAPS.git
cd DOPAPS
```

---

## Step 2: Set Up Python Environment & Install Dependencies

Setting up an isolated virtual environment keeps the project clean and avoids conflicts with other Python apps.

1. Open **Command Prompt** or **PowerShell**.
2. Navigate to your project folder:
   ```powershell
   cd "C:\path\to\your\extracted\DOPAPS"
   ```
3. Create a Virtual Environment named `.venv`:
   ```powershell
   python -m venv .venv
   ```
4. Activate the Virtual Environment:
   * **In Command Prompt (`cmd`)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **In PowerShell**:
     ```powershell
     .venv\Scripts\activate
     ```
   *(When activated, you will see `(.venv)` displayed at the start of your command prompt line).*

5. Install all required dependencies using `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```
   *This automatically installs PyQt5, PyVista, SciPy, NumPy, Scikit-Learn, and PyInstaller.*

---

## Step 3: Run the Application Locally

Once installation completes, test running the app locally:

1. Ensure `.venv` is activated.
2. Run the main application file:
   ```powershell
   python stl_viewer.py
   ```
3. The **DOPA Window** will launch immediately!
4. **Testing the App**:
   * Click **📂 Open Mesh** and select the included sample file `mandible.obj.obj`.
   * Wait a second as detected cusp tips appear as green spheres.
   * Hover over 3 green cusp tips and press **`P`** on your keyboard for each point.
   * The blue transparent **Monson Sphere** and deviation analysis report will calculate automatically.

---

## Step 4: Build a Standalone Executable (.exe)

You can turn the Python application into a standalone Windows `.exe` file using **PyInstaller**. This allows anyone to run DOPA without needing Python installed on their computer.

### Option A: Create a Portable Folder EXE (Recommended - Fast & Stable)
This creates a folder containing `DOPA.exe` along with all required 3D graphics DLLs.

1. Make sure your `.venv` is activated.
2. Run this command in your project directory:
   ```powershell
   pyinstaller --noconfirm --onedir --windowed --icon=logo.ico --name=DOPA stl_viewer.py
   ```
3. When PyInstaller finishes, open the newly created `dist\DOPA\` folder.
4. Inside `dist\DOPA\`, you will see **`DOPA.exe`**.

### Option B: Create a Single File EXE
If you prefer a single `.exe` file:
```powershell
pyinstaller --noconfirm --onefile --windowed --icon=logo.ico --name=DOPA stl_viewer.py
```
*(Note: Single-file executables take a few seconds longer to unpack into memory when double-clicked by end users).*

---

## Step 5: Distribute to Non-Technical End Users

To share the application with dentists, colleagues, or clients who do **NOT** have Python or developer tools:

1. Go to the `dist\` folder in your project directory.
2. Select the `DOPA` folder (created in Step 4 Option A).
3. (Optional) Copy sample mesh files (`mandible.obj.obj`) or `logo.ico` into the `dist\DOPA\` folder if you want them pre-packaged.
4. Right-click the `DOPA` folder → **Send to** → **Compressed (zipped) folder**.
5. Share the resulting `DOPA.zip` file via Google Drive, OneDrive, Email, or USB Flash Drive.

### Instructions for End Users:
Give the end user these 3 simple instructions:
1. **Download & Unzip** `DOPA.zip` to your desktop.
2. Open the unzipped `DOPA` folder.
3. Double-click **`DOPA.exe`** to start using the app immediately!

---

## ❓ Troubleshooting & FAQ

### Q1: I get `'python' is not recognized as an internal or external command`
* **Cause**: Python was installed without checking "Add python.exe to PATH".
* **Solution**: Re-run the Python installer, choose **Modify**, check **Add Python to PATH**, and complete installation. Then restart Command Prompt.

### Q2: Windows shows a popup saying "Windows protected your PC" (SmartScreen)
* **Cause**: Windows Defender flags newly built, unsigned `.exe` files from unknown developers.
* **Solution**: Click **More info** on the popup, then click **Run anyway**.

### Q3: How do I undo `agent.md` being committed to GitHub?
If `agent.md` was accidentally committed and pushed to your GitHub repository:
1. Untrack the file from Git history while keeping your local file safe:
   ```powershell
   git rm --cached agent.md
   ```
2. Make sure `agent.md` is listed in your `.gitignore` file.
3. Commit and push the change to GitHub:
   ```powershell
   git commit -m "Untrack agent.md and add to .gitignore"
   git push origin master
   ```

---

## 📞 Support & Maintenance
For any issues regarding mesh loading formats or algorithm parameters, refer to `stl_viewer.py` inline code documentation.
