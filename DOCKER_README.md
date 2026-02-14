# Docker Setup Guide

## Quick Start

1. **Copy environment variables:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file with your credentials:**
   ```bash
   # Required
   OPENAI_API_KEY=sk-your-openai-key-here
   
   # For Qdrant Cloud (Recommended)
   QDRANT_USE_CLOUD=true
   QDRANT_URL=https://your-cluster.region.gcp.cloud.qdrant.io:6333
   QDRANT_API_KEY=your-qdrant-cloud-key
   ```

3. **Build and run:**
   ```bash
   # Development mode (with hot reload)
   docker-compose up --build
   
   # Production mode  
   docker-compose -f docker-compose.yml --profile production up --build -d
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

## Environment Configurations

### Development with Qdrant Cloud
```env
OPENAI_API_KEY=sk-your-key
QDRANT_USE_CLOUD=true
QDRANT_URL=https://your-cluster.region.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your-cloud-key
DEBUG=true
```

### Development with Local Qdrant
```bash
# Start with local Qdrant
docker-compose --profile local-qdrant up --build

# Environment
OPENAI_API_KEY=sk-your-key
QDRANT_USE_CLOUD=false
QDRANT_HOST=qdrant
DEBUG=true
```

### Production
```env
OPENAI_API_KEY=sk-prod-key
QDRANT_USE_CLOUD=true
QDRANT_URL=https://prod-cluster.region.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=prod-cloud-key
ALLOWED_ORIGINS=https://your-app.com
DEBUG=false
```

## Docker Commands

### Build
```bash
docker build -t ocr-embedding-service .
```

### Run with Docker only
```bash
docker run -p 8000:8000 --env-file .env ocr-embedding-service
```

### Development
```bash
# Start in development mode (auto-reload)
docker-compose up --build

# View logs
docker-compose logs -f ocr-embedding-service

# Access container
docker-compose exec ocr-embedding-service bash
```

### Production
```bash
# Start in production mode (detached)
docker-compose -f docker-compose.yml up --build -d

# Scale service
docker-compose up --scale ocr-embedding-service=3 -d
```

### With Local Qdrant
```bash
# Start everything including local Qdrant
docker-compose --profile local-qdrant up --build

# Qdrant Web UI: http://localhost:6334
```

## Troubleshooting

### Check service health
```bash
curl http://localhost:8000/health
```

### View logs
```bash
docker-compose logs ocr-embedding-service
docker-compose logs qdrant  # if using local
```

### Reset everything
```bash
docker-compose down -v
docker-compose up --build
```

### Common issues

1. **OpenAI API key error:**
   - Check your `.env` file
   - Ensure `OPENAI_API_KEY` is valid

2. **Qdrant connection failed:**
   - Verify Qdrant Cloud credentials
   - Check network connectivity

3. **Port already in use:**
   - Change `SERVER_PORT` in `.env`
   - Kill existing processes: `lsof -ti:8000 | xargs kill -9`

## API Endpoints

- `POST /extract` - Extract text from images/PDFs
- `POST /upload-document` - Upload and index documents
- `POST /upload-textbook` - Upload textbooks with metadata
- `POST /upload-batch` - Batch upload documents
- `GET /health` - Health check

## Volume Mounts

- `uploads_data:/tmp/uploads` - Persistent temp file storage
- `./logs:/app/logs` - Application logs (development)
- `qdrant_data:/qdrant/storage` - Qdrant data (local mode)

## Environment Variables Reference

See `.env.example` for full list of configurable options.