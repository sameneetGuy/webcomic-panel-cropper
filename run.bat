@echo off
setlocal enabledelayedexpansion

REM Prefer Python Launcher if available
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)

REM If files were dragged onto this .bat, process those
if not "%~1"=="" (
  echo Processing dragged files...
  for %%F in (%*) do (
    echo =========================================
    echo Processing "%%~fF"
    %PY% auto_crop_panels.py "%%~fF"
  )
  goto done
)

REM Otherwise, process common image files in current folder
echo Processing images in current folder...
for %%f in (*.png *.jpg *.jpeg) do (
  echo =========================================
  echo Processing "%%f"
  %PY% auto_crop_panels.py "%%f"
)

:done
echo.
echo All done.
pause
