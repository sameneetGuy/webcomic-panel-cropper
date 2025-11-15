@echo off
setlocal

set PYTHON=python

for %%f in (*.png *.jpg *.jpeg) do (
    echo =========================================
    echo Processing %%f
    %PYTHON% auto_crop_panels.py "%%f"
)

echo.
echo All done.
pause
