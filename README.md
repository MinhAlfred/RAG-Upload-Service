# 📚 Document Embedding Service

A comprehensive OCR and document processing service that extracts text from various document formats, generates embeddings, and stores them in a vector database for AI-powered applications.

## 🌟 Features

### 📄 Document Processing
- **Multi-format support**: PDF (text + scanned), images (PNG, JPEG), text files
- **Advanced OCR**: Tesseract OCR with Vietnamese and English language support  
- **Smart text extraction**: Handles both digital and scanned documents
- **Textbook processing**: Enhanced metadata parsing for educational materials

### 🤖 AI-Powered Embeddings
- **OpenAI Integration**: Uses `text-embedding-3-small` (1536 dimensions)
- **Intelligent chunking**: Configurable chunk sizes with overlap
- **Batch processing**: Efficient handling of multiple documents
- **Semantic search ready**: Optimized for RAG applications

### 🗄️ Vector Database
- **Qdrant Integration**: Both Cloud and self-hosted options
- **Scalable storage**: Handles large document collections
- **Fast retrieval**: Optimized for similarity search
- **Metadata filtering**: Rich metadata for context-aware search

### 🚀 Production Ready
- **FastAPI**: Modern, fast web API framework
- **Docker Support**: Complete containerization with docker-compose
- **Health Monitoring**: Built-in health checks and monitoring
- **Configurable CORS**: Flexible cross-origin resource sharing

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  OCR & Extract  │───▶│   Text Chunks   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Vector Database │◀───│   Embeddings    │◀───│ OpenAI API      │
│    (Qdrant)     │    │  (1536 dims)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd ocr
   cp .env.example .env
   ```

2. **Configure environment:**
   ```bash
   # Edit .env with your credentials
   OPENAI_API_KEY=sk-your-openai-key-here
   QDRANT_URL=https://your-cluster.region.gcp.cloud.qdrant.io:6333
   QDRANT_API_KEY=your-qdrant-key
   ```

3. **Start service:**
   ```bash
   # Development mode
   docker-compose up --build
   
   # Production mode
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

4. **Access API:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health: http://localhost:8000/health

### Option 2: Local Development

1. **Prerequisites:**
   ```bash
   # Install Tesseract OCR
   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   # Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-vie
   # macOS: brew install tesseract
   ```

2. **Setup Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the service:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📚 API Documentation

### Core Endpoints

#### 1. Extract Text Only
```http
POST /extract
Content-Type: multipart/form-data

file: <document file>
```

**Response:**
```json
{
  "text": "Extracted text content...",
  "filename": "document.pdf",
  "file_type": "application/pdf",
  "char_count": 1234,
  "status": "success"
}
```

#### 2. Process & Store Document
```http
POST /upload-document
Content-Type: multipart/form-data

file: <document file>
metadata: <optional JSON string>
```

**Response:**
```json
{
  "document_id": "doc_abc123",
  "filename": "document.pdf",
  "chunks_count": 15,
  "status": "success",
  "message": "Document processed and indexed successfully"
}
```

#### 3. Process Textbook (Enhanced Metadata)
```http
POST /upload-textbook
Content-Type: multipart/form-data

file: <textbook file>
book_name: "Sách Toán"
publisher: "Cánh Diều"
grade: "Lớp 3"                    # Optional
product_name: "Custom name"       # Optional
```

**Response:**
```json
{
  "document_id": "doc_textbook_123",
  "filename": "sach_toan.pdf",
  "chunks_count": 25,
  "status": "success",
  "message": "Textbook processed successfully. Book: Sách Toán - Cánh Diều - Lớp 3"
}
```

#### 4. Batch Upload
```http
POST /upload-batch
Content-Type: multipart/form-data

files: <multiple document files>
```

#### 5. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "qdrant_connected": true
}
```

### Management Endpoints

- `DELETE /document/{document_id}` - Delete document
- `GET /document/{document_id}/metadata` - Get document metadata
- `GET /collections/{collection_name}/info` - Get collection info

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `QDRANT_URL` | Qdrant Cloud URL | - | If using cloud |
| `QDRANT_API_KEY` | Qdrant API key | - | If using cloud |
| `QDRANT_HOST` | Qdrant host | localhost | If local |
| `CHUNK_SIZE` | Text chunk size | 800 | ❌ |
| `CHUNK_OVERLAP` | Chunk overlap | 150 | ❌ |
| `MAX_FILE_SIZE_MB` | Max upload size | 100 | ❌ |
| `COLLECTION_NAME` | Qdrant collection | cs_chatbot_docs | ❌ |
| `ALLOWED_ORIGINS` | CORS origins | * | ❌ |

### Supported File Types

- **PDF**: `application/pdf`
- **Images**: `image/png`, `image/jpeg`, `image/jpg`
- **Text**: `text/plain`, `text/markdown`
- **Code**: `text/x-python`
- **JSON**: `application/json`

## 🐳 Docker Usage

### Using Makefile (Recommended)
```bash
# Setup
make setup          # Copy .env.example to .env
make dev            # Start development
make prod           # Start production
make local          # Start with local Qdrant

# Monitoring
make logs           # View logs
make health         # Check health
make status         # Container status

# Maintenance
make clean          # Clean everything
make restart        # Restart services
```

### Manual Docker Commands
```bash
# Development
docker-compose up --build

# Production
docker-compose -f docker-compose.prod.yml up --build -d

# With local Qdrant
docker-compose --profile local-qdrant up --build
```

## 🔧 Development

### Project Structure
```
ocr/
├── main.py                 # FastAPI application
├── config/
│   └── config.py          # Configuration settings
├── services/
│   ├── embedding_service.py    # Document processing
│   ├── document_processor.py   # Text extraction
│   ├── embedder.py            # OpenAI integration
│   └── qdrant_service.py      # Vector database
├── model/
│   └── schemas.py         # Pydantic models
├── requirements.txt       # Dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Development setup
└── docker-compose.prod.yml # Production setup
```

### Adding New Features

1. **New file type support:**
   - Add MIME type to `SUPPORTED_FILE_TYPES` in config
   - Implement extraction logic in `document_processor.py`

2. **Custom embeddings:**
   - Extend `embedder.py` with new provider
   - Update `embedding_service.py` initialization

3. **Enhanced metadata:**
   - Modify schemas in `model/schemas.py`
   - Update processing logic in services

### Testing
```bash
# Run tests locally
pytest

# Run tests in Docker
docker-compose exec ocr-embedding-service python -m pytest

# API testing
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/docs
```

## 🌐 Production Deployment

### 1. Environment Setup
```env
# Production .env
OPENAI_API_KEY=sk-prod-key-here
QDRANT_URL=https://prod-cluster.region.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=prod-qdrant-key
ALLOWED_ORIGINS=https://your-app.com,https://admin.your-app.com
DEBUG=false
MAX_FILE_SIZE_MB=50
```

### 2. Docker Deployment
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up --build -d

# With scaling
docker-compose -f docker-compose.prod.yml up --scale ocr-embedding-service=3 -d
```

### 3. Reverse Proxy (Nginx)
```nginx
# nginx.conf
upstream ocr_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-api.com;
    
    location / {
        proxy_pass http://ocr_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 Monitoring

### Health Checks
```bash
# Service health
curl http://localhost:8000/health

# Container health  
docker inspect --format "{{.State.Health.Status}}" ocr-embedding-api
```

### Logs
```bash
# Application logs
docker-compose logs -f ocr-embedding-service

# Specific timeframe
docker-compose logs --since 1h ocr-embedding-service
```

### Resource Monitoring
```bash
# Container stats
docker stats

# Using Makefile
make stats
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature/amazing-feature`
3. **Commit changes:** `git commit -m 'Add amazing feature'`
4. **Push to branch:** `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Code Standards
- Follow PEP 8 for Python code
- Use type hints
- Add docstrings for functions
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

1. **Tesseract not found:**
   ```bash
   # Install Tesseract OCR
   # Make sure it's in system PATH
   tesseract --version
   ```

2. **OpenAI API errors:**
   - Check API key validity
   - Verify account credits
   - Check rate limits

3. **Qdrant connection issues:**
   - Verify URL and API key
   - Check network connectivity
   - Ensure collection exists

4. **Docker issues:**
   ```bash
   # Reset everything
   docker-compose down -v
   docker system prune -f
   docker-compose up --build
   ```

### Getting Help

- 📧 **Email**: your-email@domain.com
- 💬 **Issues**: Create GitHub issue
- 📚 **Documentation**: `/docs` endpoint
- 🐳 **Docker Help**: See [DOCKER_README.md](DOCKER_README.md)

## 🎯 Roadmap

- [ ] Support for more file formats (DOCX, PPTX)
- [ ] Multiple embedding providers
- [ ] Advanced text preprocessing
- [ ] Batch processing optimization
- [ ] Web dashboard for document management
- [ ] Kubernetes deployment
- [ ] API authentication and authorization

---

**Made with ❤️ for document processing and AI applications**