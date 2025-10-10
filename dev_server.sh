#!/bin/bash
# Development server startup script
# Loads environment variables and starts the API server

cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start uvicorn
echo "Starting Signal API server..."
echo "Server will be available at http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

