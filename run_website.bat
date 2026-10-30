@echo off
title EchoSafe — Privacy-Preserving Federated Social Media Platform
echo ======================================================================
echo  Privacy-Preserving Federated Learning for Social Media Influencer
echo  Launching EchoSafe Demo Social Platform (FastAPI + React Production)...
echo ======================================================================
echo.
echo  EchoSafe Web App : http://127.0.0.1:8000/
echo  Swagger API Docs : http://127.0.0.1:8000/docs
echo  Phase II Monitor : http://127.0.0.1:8000/dashboard
echo.
echo  Opening EchoSafe in your browser...
timeout /t 2 >nul
start http://127.0.0.1:8000/
.\python.bat -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
