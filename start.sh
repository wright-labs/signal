#!/bin/bash
# Railway start script for Signal API

# Set default port if not provided
PORT=${PORT:-8000}

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port $PORT

