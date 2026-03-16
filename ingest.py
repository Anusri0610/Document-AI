import os
import glob
import json
from pathlib import Path
import PyPDF2
import easyocr

OUTPUT_FOLDER = "output"

# Initialize EasyOCR reader (loads into memory once)
# Using English, but can be expanded for multilingual support later (Phase 6)
ocr_reader = None

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        print("🤖 Initializing EasyOCR models... (this may take a moment)")
        # Reverting to 'en' only. Loading multiple custom language models ('ta') on 
        # Streamlit Cloud's limited PyTorch environment causes state_dict crashes.
        ocr_reader = easyocr.Reader(['en'])
    return ocr_reader

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            t = page.extract_text()
            if t: full_text += t + "\n"
    return full_text

def extract_text_from_image(image_path: str) -> str:
    from PIL import Image
    # Streamlit Cloud only provides 1GB of RAM. High resolution images will cause 
    # PyTorch/EasyOCR to crash the app (Out-Of-Memory)
    try:
        with Image.open(image_path) as img:
            img.thumbnail((800, 800))
            img.save(image_path)
    except Exception as e:
        print(f"Image compression failed: {e}")
        
    reader = get_ocr_reader()
    results = reader.readtext(image_path)
    # readtext returns a list of tuples: (bounding_box, text, confidence)
    extracted_text = " ".join([text for _, text, _ in results])
    
    # OCR on handwriting is often messy. We use Llama to clean it up before chunking/embedding.
    try:
        from groq import Groq
        from dotenv import load_dotenv
        import os
        
        load_dotenv(".env.txt")
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        prompt = f"The following text is from an OCR scan of a handwritten document and contains severe errors. Please reconstruct the original coherent text as accurately as possible. If it looks like a recipe (e.g., Chicken Sukka, masala, etc.), guess the missing words based on context. Output ONLY the reconstructed text, nothing else.\n\nRaw OCR Text:\n{extracted_text}"
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Failed to clean OCR text: {e}")
        return extracted_text

def process_domain_documents(domain: str) -> int:
    """
    Reads PDFs/Images from documents/{domain}/, extracts text via PyPDF2/OCR, 
    tags them with domain metadata, and saves JSON chunks to output/{domain}_chunks/
    """
    input_folder = f"documents/{domain}"
    print(f"\n📂 Scanning {input_folder}/ for documents...")
    
    input_files = []
    
    # Phase 2: Detect if file is image or PDF
    pdf_files = glob.glob(f"{input_folder}/*.pdf")
    img_files = glob.glob(f"{input_folder}/*.png") + glob.glob(f"{input_folder}/*.jpg") + glob.glob(f"{input_folder}/*.jpeg")
    
    input_files.extend(pdf_files)
    input_files.extend(img_files)
    
    if not input_files:
        print(f"⚠️ No documents found in {input_folder}/")
        return 0

    chunks_dir = Path(f"{OUTPUT_FOLDER}/{domain}_chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(input_files)} {domain} documents to process")
    
    total_chunks_created = 0
    for file_path in input_files:
        filename = Path(file_path).name
        filename_without_ext = Path(file_path).stem
        ext = Path(file_path).suffix.lower()
        
        # Check if already processed
        existing_chunks = list(chunks_dir.glob(f"{filename_without_ext}_*.json"))
        if existing_chunks:
            print(f"⏭️ Skipping {filename} - already processed ({len(existing_chunks)} chunks)")
            total_chunks_created += len(existing_chunks)
            continue
            
        print(f"📥 Processing {filename}...")
        try:
            # Route extraction based on file type
            if ext == '.pdf':
                text = extract_text_from_pdf(file_path)
            elif ext in ['.png', '.jpg', '.jpeg']:
                text = extract_text_from_image(file_path)
            else:
                print(f"⚠️ Unsupported file type: {ext}")
                continue
                
            if not text or not text.strip():
                print(f"⚠️ No text could be extracted from {filename}")
                continue
                
            # Split page into chunks
            chunk_size = 1000
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            chunk_count = 0
            for i, t_chunk in enumerate(text_chunks):
                chunk_id = f"c{i}"
                
                chunk_json = {
                    "chunk_id": chunk_id,
                    "chunk_type": "text",
                    "text": t_chunk,
                    "domain": domain, 
                    "source_document": filename_without_ext
                }
                
                chunk_key = chunks_dir / f"{filename_without_ext}_{chunk_id}.json"
                with open(chunk_key, "w", encoding="utf-8") as f:
                    json.dump(chunk_json, f, indent=2)
                    
                chunk_count += 1
                
            print(f"✅ Created {chunk_count} {domain} chunks for {filename}")
            total_chunks_created += chunk_count
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            
    return total_chunks_created

def run_all_ingestion():
    total = 0
    for domain in ["medical", "legal", "recipe"]:
        Path(f"documents/{domain}").mkdir(parents=True, exist_ok=True)
        total += process_domain_documents(domain)
    
    if total > 0:
        print(f"\n✅ Total new chunks created across all domains: {total}")
    else:
        print("\n✅ All documents are up to date.")

if __name__ == "__main__":
    run_all_ingestion()
