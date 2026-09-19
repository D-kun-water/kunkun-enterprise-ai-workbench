@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo 正在启动鲲坤科技企业 AI 协作工作台...
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import fastapi, streamlit, uvicorn" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=.venv\Scripts\python.exe"
)

"%PYTHON_EXE%" -c "import fastapi, streamlit, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo 未找到可用依赖。请先按 README 创建 .venv 并安装 requirements.txt。
    pause
    exit /b 1
)

start "鲲坤科技企业 AI 协作工作台 API" /min "%PYTHON_EXE%" scripts\run_api.py
start "鲲坤科技企业 AI 协作工作台前端" /min "%PYTHON_EXE%" scripts\run_frontend.py

echo 正在等待后端完成初始化...
for /l %%i in (1,1,20) do (
    powershell -NoProfile -Command "try { if ((Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health -TimeoutSec 1).StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 goto ready
    timeout /t 1 /nobreak >nul
)

echo 后端仍在启动，请稍后手动打开 http://127.0.0.1:8501
pause
exit /b 1

:ready
echo 后端地址：http://127.0.0.1:8000/docs
echo 前端地址：http://127.0.0.1:8501
start "" http://127.0.0.1:8501
