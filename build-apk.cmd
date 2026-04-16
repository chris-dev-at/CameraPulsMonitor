@echo off
echo Building APK...

echo Copying web version to mobile...
copy /Y index.html mobile\www\index.html
copy /Y css\style.css mobile\css\style.css
copy /Y css\icon.svg mobile\css\icon.svg
copy /Y js\app.js mobile\js\app.js

cd mobile

echo Building APK with Docker...
docker-compose build

echo Running build container...
docker-compose up --build

cd ..

if exist "mobile\output\*.apk" (
    echo.
    echo APK built successfully!
    echo Location: mobile\output\
) else (
    echo.
    echo Error: APK not found
    pause
    exit /b 1
)

echo.
echo Done!
pause