# -*- coding: utf-8 -*-
"""
Intell Audio Inference & Retrieval System
Root Entrypoint.

Delegates execution to frontend/streamlit_app.py for Streamlit launch compatibility.
"""

from frontend.streamlit_app import main

if __name__ == "__main__":
    main()
