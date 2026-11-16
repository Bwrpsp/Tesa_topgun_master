from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import admin, jobs, leaderboard
from app.config import settings

app = FastAPI(
    title="TESA TopGun 2026 - Master Evaluation Server",
    description="Central orchestrator for distributed evaluation of AI models on Raspberry Pi slaves.",
    version="1.0.0"
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(jobs.router)
app.include_router(admin.router)
app.include_router(leaderboard.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "TESA TopGun 2026 Master Server",
        "status": "healthy",
        "active_day": settings.ACTIVE_DAY,
        "docs_url": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.MASTER_HOST,
        port=settings.MASTER_PORT,
        reload=False
    )
