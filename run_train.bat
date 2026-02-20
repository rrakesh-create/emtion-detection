@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%\backend\src;%CD%\backend
echo Starting Audio Model Training...
python scripts/train_audio_resnet.py
pause
