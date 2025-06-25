#!/bin/bash
echo "🚀 Starting Visual Search API..."
echo "📚 Docs will be at: http://localhost:8000/docs"
source ../clip-env/bin/activate
uvicorn main-byom:app --host 127.0.0.1 --port 8000 --reload
