@echo off
REM Build 2 ung dung cho Windows (.exe mot file). Chay tai thu muc goc: build_windows.bat
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
cd CauA_GaussSeidel
pyinstaller --noconfirm CauA_GaussSeidel.spec
cd ..\CauB_MonteCarlo
pyinstaller --noconfirm CauB_MonteCarlo.spec
cd ..
echo Xong: CauA_GaussSeidel\dist\CauA_GaussSeidel.exe va CauB_MonteCarlo\dist\CauB_MonteCarlo.exe
pause
