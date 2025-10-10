# core/pdf_utils.py — OPTIMIZED FOR LARGE PDFs WITH OCR SUPPORT

import PyPDF2
from typing import List, Dict
import re
from PIL import Image
import io

# ==================== CONFIGURATION ====================
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
MIN_CHUNK_SIZE = 100  # Minimum viable chunk size
MAX_PAGES_PER_BATCH = 50  # Process PDFs in batches for memory efficiency
MIN_TEXT_THRESHOLD = 50  # Minimum text length to consider page as text-based
OCR_ENABLED = True  # Global OCR toggle

# Lazy load EasyOCR to avoid startup overhead
_ocr_reader = None

def get_ocr_reader():
    """Lazy initialization of OCR reader"""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print("Initializing EasyOCR (this may take a moment on first run)...")
            _ocr_reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if CUDA available
            print("OCR reader initialized")
        except ImportError:
            print("!EasyOCR not installed. OCR functionality disabled.")
            return None
        except Exception as e:
            print(f"!Failed to initialize OCR: {str(e)}")
            return None
    return _ocr_reader

def sanitize_text(text: str) -> str:
    """Remove NUL characters and other problematic bytes that PostgreSQL can't handle"""
    if not text:
        return ""
    
    # Remove NUL bytes (0x00)
    text = text.replace('\x00', '')
    
    # Remove other control characters except common ones (tab, newline, carriage return)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    return text

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR"""
    if not OCR_ENABLED:
        return ""
    
    reader = get_ocr_reader()
    if reader is None:
        return ""
    
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Perform OCR
        result = reader.readtext(image, detail=0, paragraph=True)
        
        # Join results
        text = ' '.join(result)
        return sanitize_text(text)
        
    except Exception as e:
        print(f"!OCR failed: {str(e)}")
        return ""

def has_images(page) -> bool:
    """Check if PDF page contains images"""
    try:
        if '/XObject' in page['/Resources']:
            x_objects = page['/Resources']['/XObject'].get_object()
            for obj in x_objects:
                if x_objects[obj]['/Subtype'] == '/Image':
                    return True
        return False
    except Exception:
        return False

def extract_images_from_page(page) -> List[bytes]:
    """Extract image bytes from PDF page"""
    images = []
    try:
        if '/XObject' not in page['/Resources']:
            return images
        
        x_objects = page['/Resources']['/XObject'].get_object()
        
        for obj in x_objects:
            if x_objects[obj]['/Subtype'] == '/Image':
                try:
                    size = (x_objects[obj]['/Width'], x_objects[obj]['/Height'])
                    data = x_objects[obj].get_data()
                    
                    # Skip very small images (likely icons/decorations)
                    if size[0] < 100 or size[1] < 100:
                        continue
                    
                    images.append(data)
                    
                except Exception as e:
                    print(f"!Failed to extract image: {str(e)}")
                    continue
        
    except Exception as e:
        print(f"!Image extraction error: {str(e)}")
    
    return images

def extract_text_from_pdf(pdf_path: str) -> Dict:
    """
    Extract text from PDF with optimizations for large files.
    Automatically uses OCR for image-based pages.
    Processes pages in batches to manage memory efficiently.
    """
    try:
        total_pages = 0
        all_pages_text = []
        ocr_pages_count = 0
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"!PDF has {total_pages} pages")
            
            # Process in batches for large PDFs
            for batch_start in range(0, total_pages, MAX_PAGES_PER_BATCH):
                batch_end = min(batch_start + MAX_PAGES_PER_BATCH, total_pages)
                print(f"  Processing pages {batch_start + 1}-{batch_end}...")
                
                for page_num in range(batch_start, batch_end):
                    try:
                        page = reader.pages[page_num]
                        
                        # Try to extract text normally
                        text = page.extract_text()
                        text = sanitize_text(text)
                        
                        # Check if page needs OCR (insufficient text + has images)
                        needs_ocr = len(text.strip()) < MIN_TEXT_THRESHOLD
                        
                        if needs_ocr and has_images(page):
                            print(f"!Page {page_num + 1}: Low text detected, attempting OCR...")
                            
                            # Extract images and perform OCR
                            images = extract_images_from_page(page)
                            ocr_texts = []
                            
                            for img_bytes in images:
                                ocr_text = extract_text_from_image(img_bytes)
                                if ocr_text:
                                    ocr_texts.append(ocr_text)
                            
                            if ocr_texts:
                                # Combine original text with OCR text
                                combined_text = text + "\n" + "\n".join(ocr_texts)
                                text = clean_text(combined_text)
                                ocr_pages_count += 1
                                print(f"!OCR extracted {len(text)} characters from page {page_num + 1}")
                            else:
                                print(f"!OCR found no text on page {page_num + 1}")
                        
                        # Clean text
                        text = clean_text(text)
                        
                        all_pages_text.append({
                            "page_num": page_num + 1,
                            "text": text,
                            "ocr_used": needs_ocr and len(text) > MIN_TEXT_THRESHOLD
                        })
                        
                    except Exception as e:
                        print(f"!Error extracting page {page_num + 1}: {str(e)}")
                        all_pages_text.append({
                            "page_num": page_num + 1,
                            "text": "",
                            "ocr_used": False
                        })
        
        if ocr_pages_count > 0:
            print(f"!OCR was used on {ocr_pages_count} pages")
        
        return {
            "pages": all_pages_text,
            "total_pages": total_pages,
            "ocr_pages": ocr_pages_count
        }
    
    except Exception as e:
        print(f"!Error reading PDF: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # First sanitize (remove NUL characters)
    text = sanitize_text(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', ' ', text)
    
    # Remove hyphenation at line breaks
    text = re.sub(r'-\s+', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()

def chunk_text(text_data: Dict) -> List[Dict]:
    """
    Create overlapping chunks from PDF text with intelligent splitting.
    Optimized for large documents (500-1000+ pages).
    """
    chunks = []
    pages = text_data["pages"]
    total_pages = text_data["total_pages"]
    
    print(f"!Chunking {total_pages} pages...")
    
    current_chunk = ""
    current_start_page = 1
    current_end_page = 1
    
    for page_data in pages:
        page_num = page_data["page_num"]
        page_text = page_data["text"]
        
        # Skip empty pages
        if not page_text or len(page_text) < 10:
            continue
        
        # Add page text to current chunk
        if current_chunk:
            current_chunk += " " + page_text
        else:
            current_chunk = page_text
            current_start_page = page_num
        
        current_end_page = page_num
        
        # Check if chunk is large enough
        while len(current_chunk) >= CHUNK_SIZE:
            # Find a good breaking point
            break_point = find_break_point(current_chunk, CHUNK_SIZE)
            
            # Extract chunk
            chunk_text = current_chunk[:break_point].strip()
            
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": chunk_text,
                    "start_page": current_start_page,
                    "end_page": current_end_page
                })
            
            # Keep overlap for context
            overlap_start = max(0, break_point - CHUNK_OVERLAP)
            current_chunk = current_chunk[overlap_start:]
            current_start_page = current_end_page
    
    # Add final chunk if it has content
    if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append({
            "text": current_chunk.strip(),
            "start_page": current_start_page,
            "end_page": current_end_page
        })
    
    print(f"!Created {len(chunks)} chunks from {total_pages} pages")
    
    # Log statistics
    if chunks:
        avg_chunk_size = sum(len(c["text"]) for c in chunks) / len(chunks)
        print(f"!Average chunk size: {avg_chunk_size:.0f} characters")
    else:
        print("!Warning: No chunks created! PDF may be image-only or corrupted.")
    
    return chunks

def find_break_point(text: str, target_size: int) -> int:
    """
    Find an intelligent breaking point near target_size.
    Prioritizes breaking at sentence or paragraph boundaries.
    """
    if len(text) <= target_size:
        return len(text)
    
    # Look for break points in order of preference
    search_start = max(0, target_size - 200)
    search_end = min(len(text), target_size + 200)
    search_region = text[search_start:search_end]
    
    # 1. Try to break at paragraph (double newline)
    paragraph_break = search_region.rfind('\n\n')
    if paragraph_break != -1:
        return search_start + paragraph_break
    
    # 2. Try to break at sentence (period, question mark, exclamation)
    sentence_break = max(
        search_region.rfind('. '),
        search_region.rfind('? '),
        search_region.rfind('! ')
    )
    if sentence_break != -1:
        return search_start + sentence_break + 2
    
    # 3. Try to break at newline
    newline_break = search_region.rfind('\n')
    if newline_break != -1:
        return search_start + newline_break
    
    # 4. Try to break at space
    space_break = search_region.rfind(' ')
    if space_break != -1:
        return search_start + space_break
    
    # 5. Hard break at target_size
    return target_size

def get_document_summary(text_data: Dict, max_length: int = 500) -> str:
    """
    Generate a quick summary of the document for display purposes.
    Useful for large documents.
    """
    pages = text_data["pages"]
    total_pages = text_data["total_pages"]
    
    # Get text from first few pages
    summary_text = ""
    for page_data in pages[:5]:  # First 5 pages
        summary_text += page_data["text"] + " "
        if len(summary_text) >= max_length * 2:
            break
    
    # Truncate to max_length
    if len(summary_text) > max_length:
        summary_text = summary_text[:max_length] + "..."
    
    return summary_text.strip()

def estimate_processing_time(total_pages: int, has_ocr: bool = False) -> int:
    """
    Estimate processing time in seconds based on page count.
    Helps set appropriate timeouts.
    """
    # Rough estimates:
    # - 100 pages: ~30 seconds (text-only)
    # - 500 pages: ~2.5 minutes (text-only)
    # - 1000 pages: ~5 minutes (text-only)
    # OCR adds significant overhead: ~5-10 seconds per page
    
    base_time = 10  # Base overhead
    
    if has_ocr:
        time_per_page = 7.0  # Much slower with OCR
    else:
        time_per_page = 0.3  # Seconds per page for text extraction
    
    estimated = base_time + (total_pages * time_per_page)
    
    # Add buffer for very large documents
    if total_pages > 500:
        estimated *= 1.2
    
    return int(estimated)