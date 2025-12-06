# core/tasks.py - Verify OCR Integration in Upload Processing

from celery import shared_task
from .models import Upload, Chunk
from .pdf_utils import extract_text_from_pdf, chunk_text, sanitize_text, get_ocr_engine
from .llm_config import embeddings, INDEX_NAME
from langchain_pinecone import Pinecone as PineconeVectorStore
import os
import time

@shared_task(bind=True, max_retries=3, time_limit=1800)
def process_upload(self, upload_id):
    """
    Process PDF upload with automatic PyMuPDF + PaddleOCR detection.
    """
    try:
        upload = Upload.objects.get(id=upload_id)
        print(f"Processing upload {upload_id}: {upload.filename}")
        
        # Update status
        upload.status = "processing"
        upload.save()
        
        # Extract text using the new optimized engine
        # This will use PyMuPDF for text and PaddleOCR for slides/images
        text_data = extract_text_from_pdf(upload.file.path)
        
        # Check if OCR was used
        ocr_pages = text_data.get("ocr_pages", 0)
        if ocr_pages > 0:
            print(f"PaddleOCR was used on {ocr_pages} pages")
            self.update_state(
                state='PROCESSING',
                meta={
                    'status': f'OCR processed {ocr_pages} image-based pages',
                    'ocr_pages': ocr_pages
                }
            )
        
        # Create chunks
        chunks = chunk_text(text_data)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            print(f"No chunks extracted from {upload.filename}")
            upload.status = "failed"
            upload.save()
            return {
                "success": False,
                "error": "No text extracted. File may be corrupted or empty."
            }
        
        print(f"Created {total_chunks} chunks from {text_data['total_pages']} pages")
        
        # Delete existing chunks
        Chunk.objects.filter(upload=upload).delete()
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Delete existing vectors for this file
        try:
            vector_store.delete(filter={"upload_id": upload.id})
        except Exception as e:
            print(f"Note: Could not delete vectors (might not exist): {e}")
        
        # Update metadata
        upload.pages = text_data["total_pages"]
        upload.ocr_pages = ocr_pages
        upload.ocr_used = ocr_pages > 0
        upload.save()
        
        # Process chunks in batches for Pinecone
        BATCH_SIZE = 50
        processed = 0
        
        for batch_start in range(0, total_chunks, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_texts = []
            batch_ids = []
            batch_metadatas = []
            
            for i, chunk_data in enumerate(batch_chunks):
                global_idx = batch_start + i
                
                sanitized_text = sanitize_text(chunk_data["text"])
                
                if not sanitized_text or len(sanitized_text) < 10:
                    continue
                
                chunk_id = f"{upload.id}_{global_idx}"
                
                # Save to PostgreSQL
                Chunk.objects.create(
                    upload=upload,
                    chunk_id=chunk_id,
                    text=sanitized_text,
                    start_page=chunk_data["start_page"],
                    end_page=chunk_data["end_page"]
                )
                
                # Prepare for VectorDB
                batch_texts.append(sanitized_text)
                batch_ids.append(chunk_id)
                batch_metadatas.append({
                    "upload_id": upload.id,
                    "file": upload.filename,
                    "start_page": chunk_data["start_page"],
                    "end_page": chunk_data["end_page"]
                })
            
            # Upload to Pinecone
            if batch_texts:
                try:
                    vector_store.add_texts(
                        texts=batch_texts,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                    processed += len(batch_texts)
                    print(f"Batch {batch_start//BATCH_SIZE + 1}: Uploaded {len(batch_texts)} chunks")
                    
                    # Rate limiting / Courtesy sleep
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Failed to add batch: {str(e)}")
                    raise
        
        # Mark as processed
        upload.status = "processed"
        upload.save()
        
        return {
            "success": True,
            "upload_id": upload.id,
            "total_pages": text_data["total_pages"],
            "chunks_processed": processed,
            "ocr_used": ocr_pages > 0
        }
        
    except Exception as e:
        print(f"Error processing upload {upload_id}: {str(e)}")
        try:
            upload = Upload.objects.get(id=upload_id)
            upload.status = "failed"
            upload.save()
        except:
            pass
        raise

# ==================== OCR STATUS CHECKER ====================

@shared_task
def check_ocr_availability():
    """
    Check if PaddleOCR is properly installed and configured.
    """
    try:
        print("Checking PaddleOCR availability...")
        engine = get_ocr_engine()
        
        if engine is None:
            return {
                "available": False,
                "error": "PaddleOCR failed to initialize",
                "engine": "PaddleOCR"
            }
        
        # PaddleOCR doesn't expose a simple language list property like EasyOCR,
        # but if we initialized it with lang='th', it supports TH/EN.
        return {
            "available": True,
            "engine": "PaddleOCR",
            "languages": ["en", "th"], 
            "gpu_enabled": False # Assuming CPU environment based on previous context
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }