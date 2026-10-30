"""
WebSocket Connection Manager for Real-Time Event Broadcasting.
"""

from typing import List, Dict, Any
from fastapi import WebSocket

class WebSocketManager:
    """Manages active WebSocket connections across feed, moderation, and analytics channels."""
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "feed": [],
            "moderation": [],
            "analytics": []
        }

    async def connect(self, websocket: WebSocket, channel: str = "feed"):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str = "feed"):
        if channel in self.active_connections and websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, message: Dict[str, Any], channel: str = "feed"):
        """Broadcast JSON payload to all active clients in channel."""
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception:
                    # Stale connection will be cleaned up
                    pass

ws_manager = WebSocketManager()
