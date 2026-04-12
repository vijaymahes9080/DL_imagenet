@echo off
echo 🚀 Fast-Tracking GitHub Upload...
git add .
set /p msg="Enter commit message (default: Update all project files): "
if "%msg%"=="" set msg=Update all project files
git commit -m "%msg%"
git push origin main
echo ✅ Upload Complete!
pause
