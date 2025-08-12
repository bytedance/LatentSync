@echo off
setlocal EnableExtensions

REM --- stay in script folder ---
cd /d "%~dp0"

REM --- find and load conda for plain cmd.exe ---
set "CONDA_CANDIDATES=%USERPROFILE%\miniconda3;%USERPROFILE%\anaconda3;%ProgramData%\miniconda3;%ProgramData%\anaconda3"
for %%D in (%CONDA_CANDIDATES%) do (
  if exist "%%~D\condabin\conda.bat" (
    set "CONDA_BAT=%%~D\condabin\conda.bat"
    goto :found
  )
)
echo [ERROR] Couldn't find conda.bat. Install Miniconda/Anaconda or edit the paths.
goto :end

:found
call "%CONDA_BAT%" activate base || goto :fail

REM --- env ---
call conda create -y -n latentsync python=3.10.13 || goto :fail
call conda activate latentsync || goto :fail

REM --- deps ---
python -m pip install --upgrade pip || goto :fail
call conda install -y -c conda-forge ffmpeg || goto :fail
pip install -r requirements.txt || goto :fail
pip install opencv-python-headless || goto :fail

REM --- ensure huggingface cli exists ---
pip install -U huggingface_hub || goto :fail

REM --- checkpoints (edit if your repo/files differ) ---
if not exist checkpoints mkdir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints || goto :fail
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints || goto :fail

echo [OK] Setup complete.
goto :end

:fail
echo [FAILED] Error code %errorlevel%.

:end
pause
