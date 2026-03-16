import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env.txt")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPTS = {
    "medical": """You are a medical data extraction assistant.
Extract the key medical information from the text and return it as a structured JSON object.
Use the following keys:
- "symptoms": list of strings
- "treatments": list of strings
- "medications": list of strings
- "key_findings": string summarizing the main medical point
Return ONLY valid JSON. If a category is not found, return an empty list or null.""",

    "legal": """You are a legal document parsing assistant.
Extract the key legal entities and obligations from the text and return it in a structured JSON object.
Use the following keys:
- "parties": list of strings (e.g. entities entering the contract)
- "obligations": list of strings (key duties mentioned)
- "dates": list of strings
- "summary": string summarizing the legal clause or document
Return ONLY valid JSON. If a category is not found, return an empty list or null.""",

    "recipe": """You are a culinary assistant extracting recipe data.
Read the recipe text (which may be messy from OCR) and return a clean, structured JSON object.
Use the following keys:
- "title": string (the name of the dish)
- "ingredients": list of strings (including quantities)
- "instructions": list of strings (numbered steps)
- "prep_time": string (if mentioned, otherwise null)
Return ONLY valid JSON. If a category is not found, return an empty list or null."""
}

def extract_structured_json(text: str, domain: str) -> dict:
    prompt = PROMPTS.get(domain)
    if not prompt:
        raise ValueError(f"Domain '{domain}' is not supported for structured extraction.")
        
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Text to extract from:\n{text}"}
            ],
            # Groq supports json_object forcing using response_format
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result_str = response.choices[0].message.content
        return json.loads(result_str)
        
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test the extractor
    sample_legal = "This Non-Disclosure Agreement is made carefully on March 5, 2026 by Apple Inc and Google LLC. Both parties agree not to steal secrets."
    res = extract_structured_json(sample_legal, "legal")
    print("Test Legal Extraction:", json.dumps(res, indent=2))
