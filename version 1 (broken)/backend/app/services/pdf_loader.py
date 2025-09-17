import pdfplumber

def load_pdf(file):
    """
    โหลดไฟล์ PDF จาก Django request.FILES
    คืนค่าข้อความทั้งหมดเป็น string
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text or ""  # ป้องกันกรณี page ว่าง

def split_text(text, chunk_size=1000, overlap=200):
    """
    แบ่งข้อความเป็น chunk สำหรับ LLM
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
