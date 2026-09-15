@echo off
setlocal
python "%~dp0..\dailv\dailv.py" --profile 7b %*
exit /b %ERRORLEVEL%
