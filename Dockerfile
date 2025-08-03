FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY deploy/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY perfect_prompt/ ./perfect_prompt/
COPY deploy/docker-entrypoint.sh .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
ENTRYPOINT ["./docker-entrypoint.sh"]
