# core/pdf_utils.py (PDF extraction + chunking)

import fitz
from pytesseract import image_to_string
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            full_text.append({"page_num": page_num, "text": text})
        else:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            ocr_text = image_to_string(Image.open(BytesIO(img_bytes)))
            full_text.append({"page_num": page_num, "text": ocr_text})
    total_pages = len(doc)
    doc.close()
    return {"pages": full_text, "total_pages": total_pages}

def chunk_text(text_data, chunk_size=700, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    current_text = ""
    current_start_page = None
    page_index = 0
    pages = text_data["pages"]

    while page_index < len(pages):
        page = pages[page_index]
        page_text = page["text"]
        page_num = page["page_num"]

        current_text += page_text + "\n\n"
        if current_start_page is None:
            current_start_page = page_num

        page_chunks = splitter.split_text(current_text)
        
        # ถ้ามี chunks พร้อมบันทึก
        if len(page_chunks) > 1 or len(current_text) >= chunk_size or page_index == len(pages) - 1:
            for i, chunk in enumerate(page_chunks[:-1]):  # เก็บ chunks ยกเว้นอันสุดท้าย
                chunks.append({
                    "text": chunk,
                    "start_page": current_start_page,
                    "end_page": page_num
                })
            current_text = page_chunks[-1] if page_chunks else ""
            current_start_page = page_num if current_text else None
        page_index += 1

    if current_text.strip():
        chunks.append({
            "text": current_text,
            "start_page": current_start_page,
            "end_page": page_num
        })

    return chunks