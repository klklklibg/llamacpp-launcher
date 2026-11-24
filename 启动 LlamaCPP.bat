@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem 按你的环境改 PYW，或用 PATH 里的 pythonw
set "PYW=pythonw.exe"
start "" "%PYW%" launcher.py
