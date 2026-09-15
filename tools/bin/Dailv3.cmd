@echo off
setlocal
python "%~dp0..\dailv\dailv.py" --profile 3b %*
exit /b %ERRORLEVEL%
