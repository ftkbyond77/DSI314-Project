import pandas as pd
import pdfplumber
import requests
import io
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import os
import json
import psutil
import gc

# -------------------------------
# Setup logging
# -------------------------------
os.makedirs("Extract", exist_ok=True)
logging.basicConfig(
    filename="Extract/extract_pdf.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -------------------------------
# Paths
# -------------------------------
csv_path = "Extract/data-need-extract/OTL.csv.csv"
output_csv_prefix = "Extract/e_"  # e_1.csv, e_2.csv, ...
output_json = "Extract/checkpoint.json"

# -------------------------------
# Load CSV
# -------------------------------
try:
    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
except Exception as e:
    logging.error(f"Error loading CSV: {e}")
    raise

# Filter PDF URLs
pdf_df = df[df['URL 1'].str.endswith('.pdf', na=False)].copy()
pdf_df['content_extracted'] = ""  # สร้างคอลัมน์ว่าง

# -------------------------------
# Load previous checkpoint if exists
# -------------------------------
results = {}
if os.path.exists(output_json):
    try:
        with open(output_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logging.info(f"Loaded previous checkpoint: {len(results)} PDFs already processed")
        pdf_df['content_extracted'] = pdf_df['URL 1'].map(results).fillna("")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")

# -------------------------------
# Normalize text function
# -------------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9ก-๙\s]', '', text)
    return text.strip()

# -------------------------------
# Docker-safe memory check
# -------------------------------
def get_docker_memory_usage_gb():
    """
    Returns (usage_gb, limit_gb)
    Supports both Docker and non-Docker environments
    """
    try:
        usage_path = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        limit_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(usage_path) and os.path.exists(limit_path):
            with open(usage_path, "r") as f:
                usage_bytes = int(f.read().strip())
            with open(limit_path, "r") as f:
                limit_bytes = int(f.read().strip())
            usage_gb = usage_bytes / (1024 ** 3)
            limit_gb = limit_bytes / (1024 ** 3)
            return usage_gb, limit_gb
        else:
            # fallback if outside docker
            mem = psutil.virtual_memory()
            return mem.used / (1024 ** 3), mem.total / (1024 ** 3)
    except Exception:
        mem = psutil.virtual_memory()
        return mem.used / (1024 ** 3), mem.total / (1024 ** 3)

# -------------------------------
# PDF extraction function with retry
# -------------------------------
def extract_pdf_text_from_url(url, max_retry=2):
    attempt = 0
    while attempt <= max_retry:
        try:
            url_clean = url.replace("\\", "/").strip()
            if not url_clean.startswith("http"):
                url_clean = "https://" + url_clean.lstrip("/")

            response = requests.get(url_clean, timeout=15)
            response.raise_for_status()

            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = "".join((page.extract_text() or "") + "\n" for page in pdf.pages)

            cleaned_text = normalize_text(text)
            logging.info(f"[OK] {url_clean} | {len(cleaned_text)} chars")
            return cleaned_text

        except requests.exceptions.Timeout:
            logging.warning(f"[TIMEOUT] {url} | Attempt {attempt+1}")
            attempt += 1
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            logging.error(f"[REQUEST ERROR] {url} -> {e}")
            attempt += 1
            time.sleep(5)
        except Exception as e:
            logging.error(f"[PDF ERROR] {url} -> {e}")
            attempt += 1
            time.sleep(5)

    logging.error(f"[FAILED] {url} after {max_retry+1} attempts")
    return "[ERROR] Failed to extract"

# -------------------------------
# Memory-aware batch parallel processing with append CSV
# -------------------------------
batch_size = 5
max_workers = min(2, os.cpu_count() or 1)

urls_to_process = pdf_df[pdf_df['content_extracted'].isna() | (pdf_df['content_extracted'] == "")]['URL 1'].tolist()
total = len(urls_to_process)

logging.info(f"Starting extraction: {total} PDFs to process")
file_index = 1  # สำหรับ output CSV e_1.csv, e_2.csv ...

for batch_start in range(0, total, batch_size):
    batch_urls = urls_to_process[batch_start: batch_start + batch_size]
    batch_results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_pdf_text_from_url, url): url for url in batch_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing batch {batch_start//batch_size + 1}"):
            url = futures[future]
            try:
                batch_results[url] = future.result()
            except Exception as e:
                batch_results[url] = f"[ERROR] {e}"
                logging.error(f"[FUTURE ERROR] {url} -> {e}")

            # Check Docker memory usage
            usage_gb, limit_gb = get_docker_memory_usage_gb()
            if usage_gb >= 0.9 * limit_gb:  # 90% ของ limit
                logging.warning(f"High memory usage in Docker: {usage_gb:.2f}/{limit_gb:.2f} GB, throttling...")
                time.sleep(10)

    # Merge batch results
    results.update(batch_results)
    pdf_df.loc[pdf_df['URL 1'].isin(batch_results.keys()), 'content_extracted'] = pdf_df['URL 1'].map(batch_results)

    # Save checkpoint JSON
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Checkpoint saved: {len(results)} PDFs processed")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")

    # Save batch to separate CSV e_1.csv, e_2.csv ...
    batch_df = pdf_df[pdf_df['URL 1'].isin(batch_results.keys())][['URL 1', 'content_extracted']]
    batch_file = f"{output_csv_prefix}{file_index}.csv"
    try:
        batch_df.to_csv(batch_file, index=False, encoding='utf-8-sig')
        logging.info(f"Saved batch CSV: {batch_file}")
    except Exception as e:
        logging.error(f"Error saving batch CSV {batch_file}: {e}")

    file_index += 1

    # Release memory
    del batch_results, batch_df
    gc.collect()

# -------------------------------
# Final merge and save main CSV
# -------------------------------
df = df.merge(pdf_df[['URL 1', 'content_extracted']], on='URL 1', how='left')
final_output_csv = "Extract/data/OTL_Extracted.csv"

try:
    df.to_csv(final_output_csv, index=False, encoding='utf-8-sig')
    logging.info("✅ Extraction Completed and CSV saved.")
except Exception as e:
    logging.error(f"Error saving final CSV: {e}")

print("\n✅ Extraction Completed!")
print(f"📁 Log saved to: Extract/extract_pdf.log")
print(f"📁 Partial CSV batches saved as: e_1.csv, e_2.csv, ...")
print(f"📁 Final CSV saved to: {final_output_csv}")
