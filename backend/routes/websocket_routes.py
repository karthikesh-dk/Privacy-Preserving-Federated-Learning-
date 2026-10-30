"""
WebSocket Endpoints for Real-Time Live Feeds and Moderation Alerts.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.websocket_manager import ws_manager

router = APIRouter(tags=["WebSockets"])

@router.websocket("/ws/feed")
async def websocket_feed_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket, channel="feed")
    try:
        while True:
            # Keep connection open; receive ping or client heartbeat
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel="feed")

@router.websocket("/ws/moderation")
async def websocket_moderation_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket, channel="moderation")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel="moderation")

@router.websocket("/ws/analytics")
async def websocket_analytics_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket, channel="analytics")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel="analytics")
