import json
import csv
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
from openai import OpenAI
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", 1536))
BATCH_SIZE = int(os.getenv("MAX_CHUNKS_PER_BATCH", 50))
DATA_FOLDER = "Extract/data"

# API Keys and Pinecone Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dsi314")

# ==================== DATA STRUCTURES ====================

@dataclass
class DocumentChunk:
    """Standardized chunk structure for vector DB insertion"""
    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata or {}
        }
    
    def to_pinecone_format(self) -> Dict:
        """Format for Pinecone upsert"""
        return {
            "id": self.chunk_id,
            "values": self.embedding,
            "metadata": {
                **self.metadata,
                "text": self.text
            }
        }

# ==================== TEXT CHUNKING ====================

class TextChunker:
    """Handles text chunking with token-based splitting"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
        logger.info(f"TextChunker initialized with chunk_size={chunk_size}, overlap={overlap}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        token_count = len(self.encoder.encode(text))
        logger.debug(f"Counted {token_count} tokens in text")
        return token_count
    
    def chunk_text(self, text: str, doc_id: str, metadata: Dict = None) -> List[DocumentChunk]:
        """Split text into overlapping chunks based on token count"""
        if not text or not text.strip():
            logger.warning(f"Empty text received for doc_id: {doc_id}")
            return []
        
        tokens = self.encoder.encode(text)
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoder.decode(chunk_tokens)
            
            chunk_id = f"{doc_id}_chunk_{chunk_num}"
            chunk_metadata = {
                **(metadata or {}),
                "doc_id": doc_id,
                "chunk_num": chunk_num,
                "start_token": start_idx,
                "end_token": end_idx,
                "total_tokens": len(chunk_tokens)
            }
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=chunk_metadata
            ))
            
            logger.debug(f"Created chunk {chunk_id} with {len(chunk_tokens)} tokens")
            start_idx += (self.chunk_size - self.overlap)
            chunk_num += 1
        
        logger.info(f"Created {len(chunks)} chunks for doc_id: {doc_id}")
        return chunks

# ==================== DOCUMENT NORMALIZER ====================

class DocumentNormalizer:
    """Normalizes different document structures into consistent format"""
    
    FIELD_MAPPINGS = {
        'title': ['title', 'name', 'heading', 'document_title', 'doc_name'],
        'content': ['content', 'text', 'body', 'description', 'full_text', 'abstract'],
        'author': ['author', 'authors', 'creator', 'by'],
        'category': ['category', 'type', 'subject', 'topic', 'domain'],
        'date': ['date', 'created_at', 'published_date', 'timestamp'],
        'source': ['source', 'file', 'filename', 'document_source']
    }
    
    def normalize_document(self, doc: Dict, source_file: str) -> Dict:
        """Normalize a single document to standardized format"""
        normalized = {
            'content': '',
            'metadata': {
                'source_file': source_file,
                'original_fields': list(doc.keys())
            }
        }
        
        content_text = self._extract_field(doc, 'content')
        
        if not content_text:
            content_text = self._concatenate_all_text_fields(doc)
            logger.warning(f"No content field found in document from {source_file}, concatenated text fields")
        
        normalized['content'] = content_text
        
        for std_field, possible_names in self.FIELD_MAPPINGS.items():
            if std_field != 'content':
                value = self._extract_field(doc, std_field)
                if value:
                    normalized['metadata'][std_field] = value
                    logger.debug(f"Extracted {std_field}: {value}")
        
        for key, value in doc.items():
            if key.lower() not in [name for names in self.FIELD_MAPPINGS.values() for name in names]:
                normalized['metadata'][f'original_{key}'] = str(value)
                logger.debug(f"Preserved original field {key}")
        
        logger.info(f"Normalized document from {source_file}")
        return normalized
    
    def _extract_field(self, doc: Dict, field_type: str) -> Optional[str]:
        """Extract field value using possible field name mappings"""
        possible_names = self.FIELD_MAPPINGS.get(field_type, [])
        
        for key in doc.keys():
            if key.lower() in possible_names:
                value = doc[key]
                if value and str(value).strip():
                    return str(value).strip()
        return None
    
    def _concatenate_all_text_fields(self, doc: Dict) -> str:
        """Concatenate all text fields when no content field is found"""
        text_parts = []
        for key, value in doc.items():
            if value and isinstance(value, (str, int, float)):
                if len(str(value)) > 10:
                    text_parts.append(f"{key}: {value}")
        result = "\n".join(text_parts)
        logger.debug(f"Concatenated {len(text_parts)} fields for content")
        return result

# ==================== EMBEDDING GENERATOR ====================

class EmbeddingGenerator:
    """Generates embeddings using OpenAI API"""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = EMBEDDING_MODEL):
        if not api_key:
            raise ValueError("OpenAI API key not found")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.rate_limit_delay = 0.1
        logger.info(f"EmbeddingGenerator initialized with model: {model}")
    
    def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = BATCH_SIZE) -> List[DocumentChunk]:
        """Generate embeddings for chunks in batches"""
        total_chunks = len(chunks)
        logger.info(f"Generating embeddings for {total_chunks} chunks")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=EMBEDDING_DIM
                )
                
                for j, embedding_data in enumerate(response.data):
                    batch[j].embedding = embedding_data.embedding
                    logger.debug(f"Generated embedding for chunk {batch[j].chunk_id}")
                
                logger.info(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {str(e)}")
                continue
        
        return chunks

# ==================== PINECONE UPLOADER ====================

class PineconeUploader:
    """Handles uploading chunks to Pinecone vector database"""
    
    def __init__(self, api_key: str = PINECONE_API_KEY, 
                 index_name: str = PINECONE_INDEX_NAME,
                 environment: str = PINECONE_ENVIRONMENT):
        if not api_key:
            raise ValueError("Pinecone API key not found")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.environment = environment
        self._initialize_index()
        logger.info(f"PineconeUploader initialized for index: {index_name}")
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                logger.info(f"Index '{self.index_name}' created successfully")
            
            self.index = self.pc.Index(self.index_name)
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to index '{self.index_name}' with {stats.total_vector_count} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    def upload_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """Upload chunks to Pinecone in batches"""
        if not chunks:
            logger.warning("No chunks to upload")
            return
        
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            logger.warning("No chunks with embeddings to upload")
            return
        
        total_chunks = len(valid_chunks)
        logger.info(f"Uploading {total_chunks} chunks to Pinecone")
        
        for i in range(0, total_chunks, batch_size):
            batch = valid_chunks[i:i + batch_size]
            vectors = [chunk.to_pinecone_format() for chunk in batch]
            
            try:
                self.index.upsert(vectors=vectors)
                logger.info(f"Uploaded {min(i + batch_size, total_chunks)}/{total_chunks} chunks")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error uploading batch {i}: {str(e)}")
                continue
        
        stats = self.index.describe_index_stats()
        logger.info(f"Upload complete. Total vectors in index: {stats.total_vector_count}")

# ==================== DOCUMENT PROCESSOR ====================

class DocumentProcessor:
    """Main processor that orchestrates the entire pipeline"""
    
    def __init__(self, openai_api_key: str = OPENAI_API_KEY, pinecone_api_key: str = PINECONE_API_KEY):
        if not openai_api_key or not pinecone_api_key:
            raise ValueError("API keys for OpenAI and Pinecone are required")
        
        self.normalizer = DocumentNormalizer()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator(openai_api_key)
        self.uploader = PineconeUploader(pinecone_api_key)
        self.processed_chunks = []
        logger.info("DocumentProcessor initialized")
    
    def process_json_file(self, filepath: str) -> List[DocumentChunk]:
        """Process a JSON file and return embedded chunks"""
        logger.info(f"Processing JSON file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file {filepath}: {str(e)}")
            return []
        
        if isinstance(data, dict):
            documents = [data]
        elif isinstance(data, list):
            documents = data
        else:
            logger.error(f"Unsupported JSON structure in {filepath}")
            return []
        
        source_file = os.path.basename(filepath)
        all_chunks = []
        
        for idx, doc in enumerate(documents):
            doc_id = f"{source_file.replace('.json', '')}_doc_{idx}"
            normalized = self.normalizer.normalize_document(doc, source_file)
            content = normalized['content']
            metadata = normalized['metadata']
            
            if not content:
                logger.warning(f"No content found in document {idx} from {source_file}")
                continue
            
            chunks = self.chunker.chunk_text(content, doc_id, metadata)
            all_chunks.extend(chunks)
            logger.info(f"Document {idx}: Created {len(chunks)} chunks")
        
        logger.info(f"Total chunks created from {filepath}: {len(all_chunks)}")
        
        embedded_chunks = self.embedder.generate_embeddings(all_chunks)
        return embedded_chunks
    
    def process_csv_file(self, filepath: str) -> List[DocumentChunk]:
        """Process a CSV file and return embedded chunks"""
        logger.info(f"Processing CSV file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                documents = list(reader)
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {str(e)}")
            return []
        
        source_file = os.path.basename(filepath)
        all_chunks = []
        
        for idx, doc in enumerate(documents):
            doc_id = f"{source_file.replace('.csv', '')}_doc_{idx}"
            normalized = self.normalizer.normalize_document(doc, source_file)
            content = normalized['content']
            metadata = normalized['metadata']
            
            if not content:
                logger.warning(f"No content found in document {idx} from {source_file}")
                continue
            
            chunks = self.chunker.chunk_text(content, doc_id, metadata)
            all_chunks.extend(chunks)
            logger.info(f"Document {idx}: Created {len(chunks)} chunks")
        
        logger.info(f"Total chunks created from {filepath}: {len(all_chunks)}")
        
        embedded_chunks = self.embedder.generate_embeddings(all_chunks)
        return embedded_chunks
    
    def process_directory(self, directory: str = DATA_FOLDER) -> List[DocumentChunk]:
        """Process all JSON and CSV files in a directory"""
        logger.info(f"Processing directory: {directory}")
        
        if not os.path.exists(directory):
            logger.error(f"Directory '{directory}' not found")
            return []
        
        files = [f for f in os.listdir(directory) if f.endswith(('.json', '.csv'))]
        if not files:
            logger.warning(f"No JSON or CSV files found in '{directory}'")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        all_chunks = []
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            if filename.endswith('.json'):
                chunks = self.process_json_file(filepath)
            elif filename.endswith('.csv'):
                chunks = self.process_csv_file(filepath)
            else:
                continue
            all_chunks.extend(chunks)
        
        self.processed_chunks = all_chunks
        logger.info(f"Total chunks processed from directory: {len(all_chunks)}")
        return all_chunks
    
    def upload_to_pinecone(self):
        """Upload processed chunks to Pinecone"""
        if not self.processed_chunks:
            logger.warning("No chunks to upload. Process documents first.")
            return
        
        self.uploader.upload_chunks(self.processed_chunks)
    
    def save_chunks_to_json(self, output_file: str = "embedded_chunks.json"):
        """Save processed chunks to JSON for inspection or later use"""
        output_data = [chunk.to_dict() for chunk in self.processed_chunks]
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Saved {len(output_data)} chunks to {output_file}")
        except Exception as e:
            logger.error(f"Error saving chunks to {output_file}: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        if not self.processed_chunks:
            logger.warning("No processed chunks for statistics")
            return {}
        
        total_chunks = len(self.processed_chunks)
        chunks_with_embeddings = sum(1 for c in self.processed_chunks if c.embedding)
        source_files = set(c.metadata.get('source_file', 'unknown') 
                         for c in self.processed_chunks)
        
        stats = {
            'total_chunks': total_chunks,
            'chunks_with_embeddings': chunks_with_embeddings,
            'source_files': list(source_files),
            'embedding_dimension': EMBEDDING_DIM,
            'chunk_size_tokens': CHUNK_SIZE,
            'chunk_overlap_tokens': CHUNK_OVERLAP,
            'embedding_model': EMBEDDING_MODEL
        }
        logger.info("Generated processing statistics")
        return stats

# ==================== USAGE EXAMPLE ====================

def main():
    """Main execution function"""
    logger.info(f"Starting Knowledge Base Ingestion System")
    logger.info(f"Configuration - Embedding Model: {EMBEDDING_MODEL}, "
                f"Dimension: {EMBEDDING_DIM}, "
                f"Chunk Size: {CHUNK_SIZE}, "
                f"Overlap: {CHUNK_OVERLAP}, "
                f"Pinecone Index: {PINECONE_INDEX_NAME}")
    
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        logger.error("Missing API keys")
        raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set in .env file")
    
    try:
        processor = DocumentProcessor(OPENAI_API_KEY, PINECONE_API_KEY)
        chunks = processor.process_directory(DATA_FOLDER)
        
        if not chunks:
            logger.warning("No chunks were created")
            return
        
        stats = processor.get_statistics()
        logger.info(f"Processing Summary - "
                   f"Total chunks: {stats['total_chunks']}, "
                   f"With embeddings: {stats['chunks_with_embeddings']}, "
                   f"Source files: {len(stats['source_files'])}")
        
        for file in stats['source_files']:
            logger.info(f"Processed file: {file}")
        
        processor.save_chunks_to_json()
        processor.upload_to_pinecone()
        
        logger.info("Processing complete")
        logger.info("Next steps: Review embedded_chunks.json, "
                   "Use Pinecone index for similarity search")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()