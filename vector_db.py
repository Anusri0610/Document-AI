import os
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

OUTPUT_FOLDER = "output"

def setup_chromadb_for_domain(domain: str):
    """
    Reads JSON chunks for a specific domain and indexes them into a dynamic ChromaDB collection.
    """
    print(f"\n🧠 Setting up Vector DB for domain: '{domain}'...")
    chunks_dir = Path(f"{OUTPUT_FOLDER}/{domain}_chunks")
    
    if not chunks_dir.exists():
        print(f"⚠️ Chunk directory {chunks_dir} does not exist. Run ingest.py first.")
        return None
        
    chunk_files = list(chunks_dir.glob("*.json"))
    
    if not chunk_files:
        print(f"⚠️ No chunk files found in {chunks_dir}. Did parsing succeed?")
        return None
        
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    collection_name = f"{domain}_knowledge_local"
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Check if we need to re-index
    if collection.count() >= len(chunk_files):
        print(f"✅ Loaded existing Chroma collection '{collection_name}' with {collection.count()} chunks.")
        return collection
    elif collection.count() > 0:
        print(f"⚠️ Existing collection has {collection.count()} chunks but found {len(chunk_files)} files. Recreating...")
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(name=collection_name)
    else:
        print(f"✅ Created new Chroma collection '{collection_name}'.")

    print(f"🔄 Embedding {len(chunk_files)} {domain} chunks. This might take a moment...")
    
    docs = []
    ids = []
    metadatas = []
    
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get("text", "")
            if not text.strip():
                continue
                
            chunk_id = data.get("chunk_id", cf.stem)
            
            docs.append(text)
            ids.append(f"{cf.stem}") # unique ID
            metadatas.append({
                "source_document": data.get("source_document", ""),
                "page": data.get("page", 0),
                "domain": data.get("domain", domain),
                "chunk_type": data.get("chunk_type", "text")
            })
            
    # Batch add to chroma
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        
        batch_embeddings = embed_model.encode(batch_docs).tolist()
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"   Indexed {min(i+batch_size, len(docs))}/{len(docs)} chunks...")
        
    print(f"✅ Vector DB setup complete for '{domain}' with {collection.count()} chunks.")
    return collection

def build_all_vectors():
    domains = ["medical", "legal", "recipe"]
    collections = {}
    for domain in domains:
        col = setup_chromadb_for_domain(domain)
        if col:
            collections[domain] = col
    return collections

if __name__ == "__main__":
    build_all_vectors()
