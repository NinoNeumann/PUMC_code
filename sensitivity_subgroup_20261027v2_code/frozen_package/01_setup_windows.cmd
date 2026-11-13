@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py install-r
if errorlevel 1 goto fail
echo Setup complete. Next: 02_check_and_validate_windows.cmd
pause
exit /b 0
:fail
echo Setup failed. Read the message above and WINDOWS_START_HERE.txt.
pause
exit /b 1
