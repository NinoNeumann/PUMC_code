@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe run.py all --device cuda --data-workers 2 --resume
echo Run ended. Check run\progress.json and run\logs for completion or failure.
pause
