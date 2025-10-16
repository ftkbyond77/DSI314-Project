# Student Assistant - AI-Powered Study Plan Generator

## Overview

Student Assistant is a Django-based web application that uses AI to analyze PDF documents and generate personalized study plans. The system processes uploaded PDF files, extracts text using OCR when necessary, creates vector embeddings, and uses a Large Language Model (LLM) to intelligently prioritize study materials based on content analysis and user constraints.

## 🚀 Key Features

- **Multi-PDF Processing**: Upload up to 10 PDF files simultaneously
- **OCR Support**: Automatic text extraction from image-based PDFs using EasyOCR
- **Intelligent Chunking**: Smart text segmentation with overlap for context preservation
- **Vector Search**: Pinecone-based semantic search for document retrieval
- **AI-Powered Prioritization**: LLM-driven study plan generation with detailed reasoning
- **User Constraints**: Optional prompts to customize study priorities
- **Asynchronous Processing**: Background task processing for large documents
- **Modern UI**: Clean, responsive interface built with Tailwind CSS

## 🏗️ Architecture

### Tech Stack

- **Backend**: Django 4.2+ with PostgreSQL database
- **Task Queue**: Celery with Redis broker
- **AI/ML**: OpenAI GPT-4o-mini, OpenAI text-embedding-3-large
- **Vector Database**: Pinecone for semantic search
- **PDF Processing**: PyPDF2, EasyOCR, pdf2image
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **Deployment**: Docker, Docker Compose, Gunicorn

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Django App    │    │   Background    │
│   (Upload/View) │◄──►│   (Views/Forms) │◄──►│   Tasks (Celery)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   PostgreSQL    │    │   Vector Store  │
                       │   (User Data)   │    │   (Pinecone)    │
                       └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
student_assistant/
├── core/                           # Main Django application
│   ├── models.py                   # Database models (Upload, Chunk, Plan)
│   ├── views.py                    # Web views and business logic
│   ├── tasks.py                    # Celery background tasks
│   ├── pdf_utils.py                # PDF processing and OCR
│   ├── llm_config.py               # AI model configuration
│   ├── forms.py                    # Django forms
│   ├── admin.py                    # Django admin interface
│   ├── serializers.py              # API serializers
│   ├── templates/core/             # HTML templates
│   │   ├── upload.html             # File upload interface
│   │   ├── result.html             # Study plan results
│   │   ├── login.html              # User authentication
│   │   └── register.html           # User registration
│   └── migrations/                 # Database migrations
├── student_assistant/              # Django project settings
│   ├── settings.py                 # Main configuration
│   ├── urls.py                     # URL routing
│   ├── celery.py                   # Celery configuration
│   ├── wsgi.py                     # WSGI application
│   └── asgi.py                     # ASGI application
├── Files_Test/                     # Test PDF files
├── media/                          # Uploaded files storage
├── staticfiles/                    # Static files (CSS, JS)
├── logs/                           # Application logs
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Multi-service orchestration
├── debug_chain.py                  # RAG chain testing script
├── test_upload.py                  # PDF processing test script
└── manage.py                       # Django management script
```

## 🔄 Workflow Process

### 1. User Upload
- User logs in and uploads 1-10 PDF files
- Optional constraint prompt (e.g., "Focus on exam prep")
- Files are validated and stored in database

### 2. PDF Processing (Background Task)
```
PDF File → Text Extraction → OCR (if needed) → Text Cleaning → Chunking → Vector Embeddings
```

**Detailed Steps:**
- **Text Extraction**: PyPDF2 extracts text from PDF pages
- **OCR Fallback**: EasyOCR processes image-based pages automatically
- **Text Sanitization**: Removes NUL characters and normalizes text
- **Intelligent Chunking**: Creates 1000-character chunks with 200-character overlap
- **Vector Embeddings**: OpenAI embeddings (1536 dimensions) for semantic search
- **Storage**: Chunks stored in PostgreSQL, vectors in Pinecone

### 3. Study Plan Generation
```
Vector Retrieval → Context Assembly → LLM Analysis → JSON Parsing → Priority Ranking
```

**AI Analysis Process:**
- **Document Retrieval**: MMR (Maximal Marginal Relevance) search for relevant chunks
- **Context Assembly**: Smart sampling and formatting for large documents
- **LLM Processing**: GPT-4o-mini analyzes content and generates priorities
- **JSON Extraction**: Robust parsing with fallback heuristics
- **Priority Ranking**: Detailed reasoning for each document's study order

### 4. Result Display
- Prioritized list with detailed explanations
- Statistics (total files, chunks, pages)
- Raw JSON for debugging
- Option to upload more files

## 🛠️ System Processing Details

### PDF Processing Pipeline

```python
# Text Extraction with OCR Support
def extract_text_from_pdf(pdf_path):
    - Process pages in batches (50 pages/batch)
    - Extract text using PyPDF2
    - Detect low-text pages (< 50 chars)
    - Apply OCR using EasyOCR (English/Thai)
    - Combine original + OCR text
    - Clean and normalize text
    - Return structured data with page metadata
```

### Intelligent Chunking Strategy

```python
# Smart Text Segmentation
def chunk_text(text_data):
    - Create 1000-character chunks with 200-character overlap
    - Preserve page boundaries for context
    - Find intelligent break points (sentences, paragraphs)
    - Skip empty or minimal content
    - Track start/end pages for each chunk
    - Generate unique chunk IDs
```

### Vector Search Configuration

```python
# Pinecone Vector Store Setup
- Index: "dsi314" (configurable)
- Dimension: 1536 (OpenAI text-embedding-3-large)
- Metric: Cosine similarity
- Cloud: AWS (us-east-1)
- Metadata: upload_id, filename, start_page, end_page
```

### LLM Study Plan Generation

```python
# RAG Chain Architecture
def generate_study_plan():
    - Retrieve relevant chunks using MMR search
    - Format context with intelligent sampling
    - Apply token limits (180,000 max)
    - Generate structured JSON with priorities
    - Include detailed reasoning for each document
    - Handle fallback scenarios gracefully
```

## 📊 Output Format

### Study Plan JSON Structure

```json
[
  {
    "file": "fundamentals.pdf",
    "priority": 1,
    "reason": "This document covers core principles and basic concepts that form the foundation of the subject. It covers essential terminology and introductory theories necessary for comprehending more complex ideas. Prioritizing this ensures learners build a strong base, reducing confusion in advanced topics.",
    "chunk_count": 45,
    "pages": 120
  },
  {
    "file": "advanced_topics.pdf", 
    "priority": 2,
    "reason": "This document delves into specialized subjects that require prior knowledge of basics. It explores advanced applications assuming familiarity with fundamental concepts. It should follow after mastering prerequisites to maximize its value.",
    "chunk_count": 78,
    "pages": 200
  }
]
```

### Prioritization Logic

1. **Content Analysis**: Holistic document analysis combining all chunks
2. **Fundamental First**: Core concepts rank higher than specialized topics
3. **Prerequisite Order**: Introductory content before advanced material
4. **Time Sensitivity**: Exam-related content receives higher priority
5. **User Constraints**: Custom prompts influence priority adjustments
6. **Document Weight**: Log-scale bonus for longer documents
7. **Dependency Awareness**: Considers learning dependencies between topics

## 🔧 Configuration

### Environment Variables

```bash
# Django Configuration
DJANGO_SECRET_KEY=your-secret-key
DEBUG=1
POSTGRES_DB=student_db
POSTGRES_USER=student_user
POSTGRES_PASSWORD=student_pass

# AI Services
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=dsi314
EMBEDDING_DIMENSION=1536
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1

# Celery Configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

### Key Parameters

- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters
- **Max Files**: 10 per upload
- **Max File Size**: 100MB per file
- **Processing Timeout**: 5 minutes
- **Rate Limiting**: 500 requests/minute
- **Vector Dimensions**: 1536 (OpenAI embedding)

## 🚀 Deployment

### Docker Compose Setup

```bash
# Start all services
docker-compose up -d

# Services included:
- web: Django application (Gunicorn)
- worker: Celery background tasks
- db: PostgreSQL database
- redis: Message broker and cache
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Database setup
python manage.py migrate
python manage.py collectstatic

# Start Celery worker
celery -A student_assistant worker -l info

# Start Django server
python manage.py runserver
```

## 🧪 Testing & Debugging

### Test Scripts

1. **debug_chain.py**: Test RAG chain independently
   ```bash
   python debug_chain.py
   ```

2. **test_upload.py**: Process test PDFs and generate plans
   ```bash
   python test_upload.py Files_Test/
   python test_upload.py --force  # Reprocess all files
   python test_upload.py --stats  # Show system statistics
   ```

### Key Testing Features

- File hash comparison for change detection
- Comprehensive error handling and logging
- Fallback mechanisms for failed operations
- Progress monitoring for long-running tasks
- Statistics reporting (database, vector store)

## 📈 Performance Optimizations

### Large Document Handling

- **Batch Processing**: Process PDFs in 50-page batches
- **Memory Management**: Stream processing for large files
- **Smart Sampling**: Adaptive chunk retrieval for large corpora
- **Token Management**: Intelligent context truncation
- **Rate Limiting**: API call throttling and retry logic

### Scalability Features

- **Asynchronous Processing**: Background task queue
- **Vector Batching**: Process embeddings in batches
- **Database Optimization**: Efficient queries and indexing
- **Caching**: Redis for session and task state
- **Horizontal Scaling**: Docker-based deployment

## 🔍 Monitoring & Logging

### Application Logs

- **Processing Status**: Upload, chunking, embedding progress
- **Error Tracking**: Detailed exception logging with stack traces
- **Performance Metrics**: Processing times and token usage
- **User Activity**: Upload patterns and usage statistics

### System Health

- **Database Statistics**: Upload counts, processing status
- **Vector Store Health**: Index statistics and fullness
- **Task Queue Status**: Celery worker health and backlog
- **Resource Usage**: Memory, CPU, and storage monitoring

## 🛡️ Security & Data Handling

### Data Protection

- **User Isolation**: Each user's data is completely separate
- **File Sanitization**: NUL character removal and text cleaning
- **Input Validation**: PDF file type and size restrictions
- **Authentication**: Django's built-in user management

### Privacy Considerations

- **Local Processing**: PDFs processed on your infrastructure
- **API Usage**: OpenAI and Pinecone API calls logged
- **Data Retention**: Configurable cleanup policies
- **Access Control**: Login-required for all operations

## 🔮 Future Enhancements

### Planned Features

- **Multi-language Support**: Expand OCR to more languages
- **Advanced Analytics**: Study time estimation and progress tracking
- **Collaborative Features**: Share study plans between users
- **API Endpoints**: REST API for external integrations
- **Mobile App**: Native mobile application
- **Advanced AI**: Fine-tuned models for specific domains

### Technical Improvements

- **Microservices**: Split into specialized services
- **Event Streaming**: Real-time processing updates
- **Advanced Caching**: Redis-based result caching
- **Load Balancing**: Multiple worker instances
- **Monitoring**: Prometheus/Grafana integration

## 📝 Usage Examples

### Basic Upload Flow

1. Register/Login to the system
2. Navigate to upload page
3. Select 1-10 PDF files
4. Optionally add constraint prompt
5. Click "Generate Study Plan"
6. Wait for processing (background)
7. View prioritized results

### Constraint Examples

- "Focus on exam preparation materials"
- "Prioritize data analysis and statistics topics"
- "I have 2 weeks before the final exam"
- "Emphasize foundational concepts first"

### Advanced Usage

- Use test scripts for batch processing
- Monitor logs for debugging issues
- Adjust chunk sizes for different document types
- Configure different LLM models for specific domains

---

**Student Assistant** provides an intelligent, scalable solution for automated study plan generation, combining modern AI capabilities with robust web application architecture to help students optimize their learning approach.
