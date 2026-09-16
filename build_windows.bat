@echo off
REM ============================================================
REM  Script build file .exe cho ca 2 ung dung (chay tren Windows)
REM  Cach dung: mo Command Prompt tai thu muc BaiTapLon, go: build_windows.bat
REM ============================================================

python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo === Building Cau 10A - Gauss-Seidel ===
cd CauA_GaussSeidel
pyinstaller --noconfirm --onefile --windowed --name "CauA_GaussSeidel" main.py
cd ..

echo.
echo === Building Cau 6B - Monte Carlo ===
cd CauB_MonteCarlo
pyinstaller --noconfirm --onefile --windowed --name "CauB_MonteCarlo" main.py
cd ..

echo.
echo Hoan tat! File .exe nam trong:
echo   CauA_GaussSeidel\dist\CauA_GaussSeidel.exe
echo   CauB_MonteCarlo\dist\CauB_MonteCarlo.exe
pause
