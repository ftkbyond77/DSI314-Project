# core/pdf_utils.py - OPTIMIZED FOR LARGE PDFs WITH OCR SUPPORT

import PyPDF2
from typing import List, Dict
import re
from PIL import Image
import io
import concurrent.futures

# ==================== CONFIGURATION ====================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_PAGES_PER_BATCH = 50
MIN_TEXT_THRESHOLD = 50
OCR_ENABLED = True

# OPTIMIZATION SETTINGS
# Reduced to 800 for max speed while keeping text readable
MAX_IMAGE_DIMENSION = 800  
MIN_IMAGE_DIMENSION = 150   
# Number of concurrent OCR threads (adjusted for typical container limits)
OCR_WORKERS = 4             

# Lazy load EasyOCR
_ocr_reader = None

def get_ocr_reader():
    """Lazy initialization of OCR reader"""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            print("Initializing EasyOCR (this may take a moment on first run)...")
            # gpu=False is default, explicitly set to False for CPU environments
            _ocr_reader = easyocr.Reader(['en', 'th'], gpu=False) 
            print("OCR reader initialized")
        except ImportError:
            print("EasyOCR not installed. OCR functionality disabled.")
            return None
        except Exception as e:
            print(f"Failed to initialize OCR: {str(e)}")
            return None
    return _ocr_reader

def sanitize_text(text: str) -> str:
    """Remove NUL characters and other problematic bytes"""
    if not text:
        return ""
    text = text.replace('\x00', '')
    return ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using OCR with aggressive speed optimizations:
    1. Grayscale conversion
    2. Bilinear downscaling to 800px
    3. Greedy decoding (beamWidth=1)
    """
    if not OCR_ENABLED:
        return ""
    
    reader = get_ocr_reader()
    if reader is None:
        return ""
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Optimization 1: Convert to Grayscale (faster processing)
        if image.mode != 'L':
            image = image.convert('L')
        
        # Optimization 2: Downscale large images using Bilinear (faster than Lanczos)
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            ratio = min(MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        
        # Optimization 3: Perform OCR with Greedy Decoding
        # beamWidth=1 disables beam search, significantly speeding up CPU inference
        # workers=0 prevents nested multiprocessing overhead
        result = reader.readtext(
            image, 
            detail=0, 
            paragraph=True, 
            beamWidth=1, 
            workers=0
        )
        
        return sanitize_text(' '.join(result))
        
    except Exception as e:
        print(f"OCR failed: {str(e)}")
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
    """Extract image bytes from PDF page with filtering"""
    images = []
    try:
        if '/XObject' not in page['/Resources']:
            return images
        
        x_objects = page['/Resources']['/XObject'].get_object()
        
        for obj in x_objects:
            if x_objects[obj]['/Subtype'] == '/Image':
                try:
                    # Filter out very small images (icons, lines, decorations)
                    width = x_objects[obj].get('/Width', 0)
                    height = x_objects[obj].get('/Height', 0)
                    
                    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                        continue
                        
                    data = x_objects[obj].get_data()
                    images.append(data)
                    
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Image extraction error: {str(e)}")
    
    return images

def process_page_ocr(page_num: int, existing_text: str, images: List[bytes]) -> Dict:
    """
    Helper function to run in thread pool.
    Performs OCR on a list of images and merges with existing text.
    """
    ocr_text_parts = []
    for img_bytes in images:
        text = extract_text_from_image(img_bytes)
        if text:
            ocr_text_parts.append(text)
    
    final_text = existing_text
    if ocr_text_parts:
        final_text += "\n" + "\n".join(ocr_text_parts)
    
    final_text = clean_text(final_text)
    
    return {
        "page_num": page_num,
        "text": final_text,
        "ocr_used": len(ocr_text_parts) > 0
    }

def extract_text_from_pdf(pdf_path: str) -> Dict:
    """
    Extract text from PDF with PARALLEL OCR processing.
    """
    try:
        all_pages_text = []
        ocr_pages_count = 0
        total_pages = 0
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"PDF has {total_pages} pages")
            
            # Process in batches
            for batch_start in range(0, total_pages, MAX_PAGES_PER_BATCH):
                batch_end = min(batch_start + MAX_PAGES_PER_BATCH, total_pages)
                print(f"Processing batch {batch_start + 1}-{batch_end}...")
                
                # Step 1: Pre-read pages and extract images in Main Thread (I/O Bound)
                tasks_to_process = []
                
                for page_num in range(batch_start, batch_end):
                    try:
                        page = reader.pages[page_num]
                        text = sanitize_text(page.extract_text() or "")
                        
                        # Determine if OCR is needed
                        # Condition: Text is short AND page has images
                        if len(text.strip()) < MIN_TEXT_THRESHOLD and has_images(page):
                            images = extract_images_from_page(page)
                            if images:
                                # Queue for parallel OCR
                                tasks_to_process.append((page_num + 1, text, images))
                            else:
                                # Low text but no extractable images found
                                all_pages_text.append({
                                    "page_num": page_num + 1,
                                    "text": clean_text(text),
                                    "ocr_used": False
                                })
                        else:
                            # Standard text page
                            all_pages_text.append({
                                "page_num": page_num + 1,
                                "text": clean_text(text),
                                "ocr_used": False
                            })
                            
                    except Exception as e:
                        print(f"Error reading page {page_num + 1}: {e}")
                        all_pages_text.append({"page_num": page_num + 1, "text": "", "ocr_used": False})

                # Step 2: Execute OCR in Parallel (CPU Bound)
                if tasks_to_process:
                    print(f"Running parallel OCR on {len(tasks_to_process)} pages with {OCR_WORKERS} threads...")
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
                        # Submit tasks
                        future_to_page = {
                            executor.submit(process_page_ocr, p_num, txt, imgs): p_num 
                            for p_num, txt, imgs in tasks_to_process
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_page):
                            try:
                                result = future.result()
                                all_pages_text.append(result)
                                if result["ocr_used"]:
                                    ocr_pages_count += 1
                            except Exception as exc:
                                print(f"OCR thread exception: {exc}")
                                # Fallback for failed page
                                p_num = future_to_page[future]
                                all_pages_text.append({
                                    "page_num": p_num, "text": "", "ocr_used": False
                                })

        # Sort results by page number to restore order
        all_pages_text.sort(key=lambda x: x["page_num"])
        
        if ocr_pages_count > 0:
            print(f"Total OCR processed pages: {ocr_pages_count}")
        
        return {
            "pages": all_pages_text,
            "total_pages": total_pages,
            "ocr_pages": ocr_pages_count
        }
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""
    text = sanitize_text(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', ' ', text)
    text = re.sub(r'-\s+', '', text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    return text.strip()

def chunk_text(text_data: Dict) -> List[Dict]:
    """Create overlapping chunks from PDF text"""
    chunks = []
    pages = text_data["pages"]
    
    current_chunk = ""
    current_start_page = 1
    current_end_page = 1
    
    for page_data in pages:
        page_num = page_data["page_num"]
        page_text = page_data["text"]
        
        if not page_text or len(page_text) < 10:
            continue
        
        if current_chunk:
            current_chunk += " " + page_text
        else:
            current_chunk = page_text
            current_start_page = page_num
        
        current_end_page = page_num
        
        while len(current_chunk) >= CHUNK_SIZE:
            break_point = find_break_point(current_chunk, CHUNK_SIZE)
            chunk_text = current_chunk[:break_point].strip()
            
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": chunk_text,
                    "start_page": current_start_page,
                    "end_page": current_end_page
                })
            
            overlap_start = max(0, break_point - CHUNK_OVERLAP)
            current_chunk = current_chunk[overlap_start:]
            current_start_page = current_end_page # Approximate
    
    if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append({
            "text": current_chunk.strip(),
            "start_page": current_start_page,
            "end_page": current_end_page
        })
    
    return chunks

def find_break_point(text: str, target_size: int) -> int:
    """Find smart break point"""
    if len(text) <= target_size:
        return len(text)
    
    search_start = max(0, target_size - 200)
    search_end = min(len(text), target_size + 200)
    region = text[search_start:search_end]
    
    # Priority: Paragraph -> Sentence -> Newline -> Space
    for pattern, offset in [('\n\n', 0), ('. ', 2), ('? ', 2), ('! ', 2), ('\n', 0), (' ', 0)]:
        idx = region.rfind(pattern)
        if idx != -1:
            return search_start + idx + offset
            
    return target_size