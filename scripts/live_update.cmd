@echo off
REM Live update: detect oil slicks in new Sentinel-1 scenes over Indian seas.
REM Scenes already processed by the current model are skipped, so this is cheap to run often.
cd /d "C:\Users\itanm\OneDrive\Desktop\sih"
"C:\Users\itanm\.venvs\oilspill\Scripts\python.exe" scripts\run_batch.py --max-scenes 500 >> "C:\oilspill-data\live_update.log" 2>&1
