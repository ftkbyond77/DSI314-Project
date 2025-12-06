# core/pdf_utils.py - OPTIMIZED FOR SPEED (PyMuPDF + PaddleOCR)

import fitz  # PyMuPDF
import re
import logging
import numpy as np
import concurrent.futures
from typing import List, Dict, Optional

# ==================== CONFIGURATION ====================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MIN_TEXT_THRESHOLD = 50  # If text chars < this, we force OCR
OCR_WORKERS = 2  # Keep low to avoid memory overflow on CPU
MAX_IMAGE_DIMENSION = 2000 # Max dimension for OCR processing

THAI_REC_MODEL_PATH = "../ocr_th_model"

# Setup Logging
logger = logging.getLogger(__name__)

# Global OCR Instance (Lazy Loaded)
_ocr_engine = None
_ocr_init_failed = False # Prevent repeated failed attempts

def get_ocr_engine():
    """Lazy initialization of PaddleOCR with specific Thai support"""
    global _ocr_engine, _ocr_init_failed
    
    if _ocr_init_failed:
        return None

    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            print(f"Initializing PaddleOCR (Using local Thai model from: {THAI_REC_MODEL_PATH})...")
            
            _ocr_engine = PaddleOCR(
                lang="en",                                
                ocr_version="PP-OCRv5",                  
                text_recognition_model_dir=THAI_REC_MODEL_PATH,   
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
            )
            print("PaddleOCR initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {str(e)}")
            _ocr_init_failed = True
            return None
    return _ocr_engine

def sanitize_text(text: str) -> str:
    """Remove NUL characters and cleanup"""
    if not text:
        return ""
    text = text.replace('\x00', '')
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t\r")

def clean_text(text: str) -> str:
    """Normalize whitespace and formatting"""
    if not text:
        return ""
    text = sanitize_text(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)
    return text.strip()

def run_paddle_ocr(image_array: np.ndarray) -> str:
    """Run PaddleOCR on a numpy image array"""
    engine = get_ocr_engine()
    if engine is None:
        return ""
    
    try:
        # result structure: [[[[box], [text, score]], ...]]
        result = engine.ocr(image_array, cls=True)
        
        extracted_lines = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                # Filter low confidence trash if needed, e.g. if line[1][1] > 0.5
                extracted_lines.append(text)
        
        return " ".join(extracted_lines)
    except Exception as e:
        logger.error(f"PaddleOCR execution failed: {e}")
        return ""

def process_page(page_args) -> Dict:
    """Worker function to process a single PDF page"""
    page_num, pdf_path = page_args
    
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        
        # 1. Try direct text extraction (Fastest)
        raw_text = page.get_text()
        clean_raw_text = clean_text(raw_text)
        
        ocr_used = False
        final_text = clean_raw_text

        # 2. Check heuristics: If text is sparse, it's likely a slide/image
        if len(clean_raw_text) < MIN_TEXT_THRESHOLD:
            # Render page to image
            # matrix=fitz.Matrix(2, 2) = 2x zoom (approx 144 DPI) for speed/quality balance
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # Convert to Numpy format for Paddle
            if pix.n < 3:
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                import cv2
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            else:
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Run OCR
            ocr_text = run_paddle_ocr(img_data)
            
            # If OCR found more text than the PDF parser, use it
            if ocr_text and len(ocr_text) > len(clean_raw_text):
                final_text = clean_text(ocr_text)
                ocr_used = True
        
        doc.close()
        
        return {
            "page_num": page_num + 1,
            "text": final_text,
            "ocr_used": ocr_used
        }
        
    except Exception as e:
        print(f"Error processing page {page_num + 1}: {e}")
        return {"page_num": page_num + 1, "text": "", "ocr_used": False}

def extract_text_from_pdf(pdf_path: str) -> Dict:
    """Main extraction function with parallel processing"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        print(f"Starting extraction for {pdf_path} ({total_pages} pages)")
        
        all_pages_text = []
        ocr_pages_count = 0
        
        tasks = [(i, pdf_path) for i in range(total_pages)]
        
        # ThreadPoolExecutor works well here because Paddle releases GIL during C++ inference
        with concurrent.futures.ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
            results = list(executor.map(process_page, tasks))
            
            for res in results:
                all_pages_text.append(res)
                if res["ocr_used"]:
                    ocr_pages_count += 1
        
        if ocr_pages_count > 0:
            print(f"OCR was triggered on {ocr_pages_count} pages.")
            
        return {
            "pages": all_pages_text,
            "total_pages": total_pages,
            "ocr_pages": ocr_pages_count
        }

    except Exception as e:
        print(f"Critical error in PDF extraction: {str(e)}")
        raise

def chunk_text(text_data: Dict) -> List[Dict]:
    """Create chunks from extracted text (Logic unchanged)"""
    chunks = []
    pages = sorted(text_data["pages"], key=lambda x: x["page_num"])
    
    current_chunk = ""
    current_start_page = 1
    current_end_page = 1
    
    for page in pages:
        page_num = page["page_num"]
        text = page["text"]
        
        if not text:
            continue
            
        if not current_chunk:
            current_start_page = page_num
        
        current_chunk += " " + text
        current_end_page = page_num
        
        while len(current_chunk) >= CHUNK_SIZE:
            split_idx = find_break_point(current_chunk, CHUNK_SIZE)
            chunk_content = current_chunk[:split_idx].strip()
            
            if len(chunk_content) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": chunk_content,
                    "start_page": current_start_page,
                    "end_page": current_end_page
                })
            
            overlap_idx = max(0, split_idx - CHUNK_OVERLAP)
            current_chunk = current_chunk[overlap_idx:]
            current_start_page = current_end_page
            
    if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append({
            "text": current_chunk.strip(),
            "start_page": current_start_page,
            "end_page": current_end_page
        })
        
    return chunks

def find_break_point(text: str, target: int) -> int:
    """Helper to find smart split point"""
    if len(text) <= target:
        return len(text)
    window_start = max(0, target - 100)
    window_end = min(len(text), target + 100)
    window = text[window_start:window_end]
    for marker in ['. ', '? ', '! ', '\n']:
        pos = window.rfind(marker)
        if pos != -1:
            return window_start + pos + 1
    pos = window.rfind(' ')
    if pos != -1:
        return window_start + pos
    return target