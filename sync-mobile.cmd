@echo off
echo Copying web version to mobile directory...

copy /Y index.html mobile\index.html
copy /Y css\style.css mobile\css\style.css
copy /Y css\icon.svg mobile\css\icon.svg
copy /Y js\app.js mobile\js\app.js
copy /Y index.html mobile\www\index.html
copy /Y css\style.css mobile\www\css\style.css
copy /Y css\icon.svg mobile\www\css\icon.svg
copy /Y js\app.js mobile\www\js\app.js

echo Done!
pause