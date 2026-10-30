@echo off
title Privacy-Preserving Federated Learning Dashboard
echo ======================================================================
echo  Privacy-Preserving Federated Learning for Social Media Influencer
echo  Launching Interactive Web Dashboard & FastAPI Backend...
echo ======================================================================
echo.
echo  Dashboard URL : http://127.0.0.1:8000/dashboard
echo  Swagger API   : http://127.0.0.1:8000/docs
echo.
echo  Opening browser in 2 seconds...
timeout /t 2 >nul
start http://127.0.0.1:8000/dashboard
.\python.bat -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
