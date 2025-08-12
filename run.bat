@echo off
setlocal EnableExtensions

REM --- go to script folder ---
cd /d "%~dp0"

REM --- locate conda ---
set "CONDA_CANDIDATES=%USERPROFILE%\miniconda3;%USERPROFILE%\anaconda3;%ProgramData%\miniconda3;%ProgramData%\anaconda3"
for %%D in (%CONDA_CANDIDATES%) do (
  if exist "%%~D\condabin\conda.bat" (
    set "CONDA_BAT=%%~D\condabin\conda.bat"
    goto :found
  )
)
echo [ERROR] Couldn't find conda.bat. Edit paths above.
exit /b 1

:found
call "%CONDA_BAT%" activate latentsync || exit /b 1

REM --- run your Gradio app script ---
python gradio_app.py
