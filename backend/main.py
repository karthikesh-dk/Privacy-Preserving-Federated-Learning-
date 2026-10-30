"""
EchoSafe FastAPI Application Entrypoint.
Privacy-Preserving Federated Social Media Platform with Live Moderation Pipeline.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import engine, Base
from backend.routes import (
    auth_routes,
    post_routes,
    comment_routes,
    influencer_routes,
    moderation_routes,
    federated_routes,
    websocket_routes
)

# Initialize database schema if not present
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EchoSafe API",
    description="Privacy-Preserving Federated Learning Social Platform with Real-Time Comment Moderation",
    version="3.0.0"
)

# CORS Middleware for React frontend (Vite port 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(auth_routes.router)
app.include_router(post_routes.router)
app.include_router(comment_routes.router)
app.include_router(influencer_routes.router)
app.include_router(moderation_routes.router)
app.include_router(federated_routes.router)
app.include_router(websocket_routes.router)

# Mount Static Frontend build if present
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def index():
    """Serves compiled React frontend if built, or fallback dashboard."""
    dist_index = os.path.join(frontend_dist, "index.html")
    if os.path.exists(dist_index):
        with open(dist_index, "r", encoding="utf-8") as f:
            return f.read()

    template_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    return "<h1>EchoSafe API Online. Visit /docs for Swagger UI.</h1>"

@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "app_name": "EchoSafe",
        "federated_learning": "Online (Flower-compatible)",
        "database": "SQLite (echosafe.db)",
        "models": "CommentClassifierMLP (1200->64->32->4)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
