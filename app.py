# -*- coding: utf-8 -*-
"""
Intell Audio Inference & Retrieval System (V3)
Root Entrypoint.

Starts the FastAPI backend application serving the API and supporting the React/TypeScript frontend.
For legacy Streamlit dev UI, run: streamlit run frontend/streamlit_app.py
"""

import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.APP_ENV == "development",
    )
