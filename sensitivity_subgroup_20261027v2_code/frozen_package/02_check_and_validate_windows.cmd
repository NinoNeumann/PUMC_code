@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe run.py check --device cuda
if errorlevel 1 goto fail
.venv\Scripts\python.exe code\cacox_cpu_validate.py --neural-device cuda --output validation\cuda_target
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py init --device cuda
if errorlevel 1 goto fail
.venv\Scripts\python.exe run.py subgroups
if errorlevel 1 goto fail
echo Validation and initialization complete. Next: 03_run_windows.cmd
pause
exit /b 0
:fail
echo Check failed. Do not start full training before resolving this failure.
pause
exit /b 1
