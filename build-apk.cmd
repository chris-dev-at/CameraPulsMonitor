@echo off
echo Building APK...

echo Copying web version to mobile...
copy /Y index.html mobile\www\index.html
copy /Y css\style.css mobile\css\style.css
copy /Y css\icon.svg mobile\css\icon.svg
copy /Y js\app.js mobile\js\app.js

echo Building Docker image...
docker build -t pulse-monitor-build mobile

echo Running container...
docker create --name get-apk2 pulse-monitor-build

echo Copying APK out...
docker cp get-apk2:/output/. mobile/output/
docker rm get-apk2

copy /Y mobile\output\app-debug.apk mobile\PulseMonitor.apk

if exist "mobile\PulseMonitor.apk" (
    echo.
    echo APK built successfully!
    echo Location: mobile\PulseMonitor.apk
) else (
    echo.
    echo Error: APK not found
    pause
    exit /b 1
)

echo.
echo Done!
pause