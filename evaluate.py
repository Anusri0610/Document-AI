import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env.txt")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Mock Ground Truth Dataset for Phase 4 Requirements
GROUND_TRUTH = {
    "medical": [
        {
            "question": "What are the common symptoms of a cold?",
            "expected_keywords": ["sneeze", "cough", "sore throat", "runny nose"]
        },
        {
            "question": "Is Vitamin C effective for preventing a cold?",
            "expected_keywords": ["vitamin c", "prevent", "evidence", "trial"]
        }
    ],
    "legal": [
        {
            "question": "Who are the parties in the dummy contract?",
            "expected_keywords": ["apple", "google", "inc", "llc"]
        },
        {
            "question": "How long does the confidentiality term last?",
            "expected_keywords": ["five", "5", "years"]
        }
    ],
    "recipe": [
        {
            "question": "What ingredients are in Grandma's Chicken Soup?",
            "expected_keywords": ["chicken", "carrots", "celery", "onion"]
        },
        {
            "question": "How long should the soup simmer?",
            "expected_keywords": ["2", "two", "hours"]
        }
    ]
}

def retrieve_context(query: str, domain: str, filename: str = None) -> str:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    collection_name = f"{domain}_knowledge_local"
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except ValueError:
        return ""
        
    query_embedding = embed_model.encode([query]).tolist()
    
    # Filter by specific file if provided, otherwise search the whole domain
    if filename:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            where={"source_document": filename}
        )
    else:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
    
    if not results or not results.get("documents") or len(results["documents"]) == 0 or len(results["documents"][0]) == 0:
        return ""
        
    return "\n\n".join(results["documents"][0])

def ask_groq(question: str, context: str) -> str:
    prompt = f"Answer the question based only on the context.\n\nContext: {context}\n\nQuestion: {question}"
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def calculate_recall(answer: str, expected_keywords: list) -> float:
    if not expected_keywords: return 0.0
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    return matches / len(expected_keywords)

def run_evaluation():
    print("="*60)
    print("🚀 Running Domain Accuracy Analysis...")
    print("="*60)
    
    results_summary = {}
    
    for domain, qa_pairs in GROUND_TRUTH.items():
        print(f"\nEvaluating Domain: [{domain.upper()}]")
        total_recall = 0.0
        
        for idx, qa in enumerate(qa_pairs):
            q = qa["question"]
            expected = qa["expected_keywords"]
            
            context = retrieve_context(q, domain)
            answer = ask_groq(q, context) if context else ""
            
            recall = calculate_recall(answer, expected)
            total_recall += recall
            
            print(f"  Q{idx+1}: {q}")
            print(f"  -> Recall: {recall*100:.1f}%")
            
        avg_recall = (total_recall / len(qa_pairs)) * 100 if qa_pairs else 0
        results_summary[domain] = avg_recall
        print(f"  [Average Accuracy for {domain.upper()}: {avg_recall:.1f}%]")
        
    print("\n" + "="*60)
    print("📊 FINAL DOMAIN COMPARISON TABLE")
    print("="*60)
    print(f"{'Domain':<15} | {'Average Recall Accuracy':<25}")
    print("-" * 45)
    for domain, score in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"{domain.upper():<15} | {score:>24.1f}%")
        
    best_domain = max(results_summary, key=results_summary.get)
    print(f"\n🏆 Best Performing Domain: {best_domain.upper()}")

if __name__ == "__main__":
    run_evaluation()
