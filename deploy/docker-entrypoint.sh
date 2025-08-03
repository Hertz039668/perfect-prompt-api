#!/bin/bash
set -e

echo "🚀 Starting Perfect Prompt API Server..."

# Download NLTK data if needed
python -c "
import nltk
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print('📥 Downloading NLTK data...')
    nltk.download('vader_lexicon', quiet=True)
"

# Start the server
exec uvicorn perfect_prompt.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --access-log \
    --log-level info
