# core/tasks.py - Verify OCR Integration in Upload Processing

from celery import shared_task
from .models import Upload, Chunk
from .pdf_utils import extract_text_from_pdf, chunk_text, sanitize_text
from .llm_config import embeddings, INDEX_NAME
from langchain_pinecone import PineconeVectorStore
import os
import time

@shared_task(bind=True, max_retries=3, time_limit=1800)  # 30 min for large files with OCR
def process_upload(self, upload_id):
    """
    Process PDF upload with automatic OCR detection.
    OCR is automatically applied to image-based pages.
    """
    try:
        upload = Upload.objects.get(id=upload_id)
        print(f"📄 Processing upload {upload_id}: {upload.filename}")
        
        # Update status
        upload.status = "processing"
        upload.save()
        
        # Extract text with automatic OCR (already built into pdf_utils.py)
        # This automatically detects and processes image-based pages
        text_data = extract_text_from_pdf(upload.file.path)
        
        # Check if OCR was used
        ocr_pages = text_data.get("ocr_pages", 0)
        if ocr_pages > 0:
            print(f"✅ OCR was used on {ocr_pages} pages (en + th)")
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
            print(f"⚠️ No chunks extracted from {upload.filename}")
            upload.status = "failed"
            upload.save()
            return {
                "success": False,
                "error": "No text extracted. File may be corrupted or empty."
            }
        
        print(f"📦 Created {total_chunks} chunks from {text_data['total_pages']} pages")
        
        # Delete existing chunks
        Chunk.objects.filter(upload=upload).delete()
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Delete existing vectors
        try:
            vector_store.delete(filter={"upload_id": upload.id})
            print(f"🗑️ Deleted existing vectors for upload {upload.id}")
        except Exception as e:
            print(f"⚠️ Could not delete vectors: {str(e)}")
        
        # Update metadata with OCR info
        upload.pages = text_data["total_pages"]
        upload.ocr_pages = ocr_pages
        upload.ocr_used = ocr_pages > 0
        upload.save()
        
        # Process chunks in batches
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
                
                # Sanitize text (removes NUL characters for PostgreSQL)
                # RENAMED: chunk_text → sanitized_text to avoid shadowing the function
                sanitized_text = sanitize_text(chunk_data["text"])
                
                if not sanitized_text or len(sanitized_text) < 10:
                    print(f"⚠️ Skipping empty chunk {global_idx}")
                    continue
                
                chunk_id = f"{upload.id}_{global_idx}"
                
                # Save to database
                try:
                    Chunk.objects.create(
                        upload=upload,
                        chunk_id=chunk_id,
                        text=sanitized_text,
                        start_page=chunk_data["start_page"],
                        end_page=chunk_data["end_page"]
                    )
                except ValueError as e:
                    if "NUL" in str(e) or "0x00" in str(e):
                        # Aggressive sanitization
                        sanitized_text = ''.join(
                            c for c in sanitized_text 
                            if c.isprintable() or c in '\t\n\r'
                        )
                        sanitized_text = sanitized_text.replace('\x00', '')
                        
                        if sanitized_text and len(sanitized_text) >= 10:
                            Chunk.objects.create(
                                upload=upload,
                                chunk_id=chunk_id,
                                text=sanitized_text,
                                start_page=chunk_data["start_page"],
                                end_page=chunk_data["end_page"]
                            )
                        else:
                            continue
                    else:
                        raise
                
                # Prepare for vector store
                batch_texts.append(sanitized_text)
                batch_ids.append(chunk_id)
                batch_metadatas.append({
                    "upload_id": upload.id,
                    "file": upload.filename,
                    "start_page": chunk_data["start_page"],
                    "end_page": chunk_data["end_page"]
                })
            
            # Add to vector store
            if batch_texts:
                try:
                    vector_store.add_texts(
                        texts=batch_texts,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                    processed += len(batch_texts)
                    print(f"✅ Batch {batch_start//BATCH_SIZE + 1}: {len(batch_texts)} chunks ({processed}/{total_chunks})")
                    
                    # Update progress
                    self.update_state(
                        state='PROCESSING',
                        meta={
                            'status': f'Embedding chunks: {processed}/{total_chunks}',
                            'progress': int((processed / total_chunks) * 100)
                        }
                    )
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Failed to add batch: {str(e)}")
                    raise
        
        # Mark as processed
        upload.status = "processed"
        upload.save()
        
        result = {
            "success": True,
            "upload_id": upload.id,
            "filename": upload.filename,
            "total_pages": text_data["total_pages"],
            "total_chunks": processed,
            "ocr_pages": ocr_pages,
            "ocr_used": ocr_pages > 0
        }
        
        print(f"✅ Successfully processed {upload.filename}")
        print(f"   Pages: {text_data['total_pages']}, Chunks: {processed}, OCR: {ocr_pages} pages")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing upload {upload_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        try:
            upload = Upload.objects.get(id=upload_id)
            upload.status = "failed"
            upload.save()
        except:
            pass
        
        # Retry logic
        if self.request.retries < self.max_retries:
            print(f"🔄 Retrying ({self.request.retries + 1}/{self.max_retries})...")
            raise self.retry(exc=e, countdown=10 * (2 ** self.request.retries))
        
        raise


# ==================== OCR STATUS CHECKER ====================

@shared_task
def check_ocr_availability():
    """
    Check if EasyOCR is properly installed and configured.
    Run this task to verify OCR setup.
    """
    try:
        from .pdf_utils import get_ocr_reader
        
        print("🔍 Checking OCR availability...")
        
        reader = get_ocr_reader()
        
        if reader is None:
            return {
                "available": False,
                "error": "EasyOCR not installed or failed to initialize",
                "languages": []
            }
        
        # Get supported languages
        languages = reader.lang_list
        
        print(f"✅ OCR is available with languages: {languages}")
        
        return {
            "available": True,
            "languages": languages,
            "gpu_enabled": reader.gpu
        }
        
    except Exception as e:
        print(f"❌ OCR check failed: {str(e)}")
        return {
            "available": False,
            "error": str(e),
            "languages": []
        }