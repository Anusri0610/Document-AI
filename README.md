# Document-AI
 A Multi-Domain Retrieval-Augmented Generation (RAG) web application with OCR capabilities, automated document classification, and structured JSON extraction using Llama-3 and Streamlit.
 
# Multi-Domain AI Assistant (Medical, Legal, Culinary)

A powerful, multimodal Retrieval-Augmented Generation (RAG) backend and interactive Streamlit frontend that dynamically processes Medical PDFs, Legal Contracts, and Handwritten Recipes (via OCR).

## 🌟 Key Features

*   **Multi-Domain Intelligence**: Automatically routes, indexes, and chats with documents across different domains using isolated Vector Database namespaces to prevent hallucination cross-contamination.
*   **Auto-Classification**: Upload an unknown file and the agent automatically sniffs the text to classify it as Legal, Medical, or Recipe.
*   **Multimodal (OCR)**: Uses `EasyOCR` to read messy handwritten recipe notes (.png, .jpg) and converts them into searchable embeddings.
*   **Structured JSON Extraction**: Utilize domain-specific Groq extraction prompts to instantly pull structured data (like Ingredients, Legal Obligations, or Cold Symptoms) into downloadable JSON files.
*   **Interactive UI**: A clean Streamlit application that handles File Uploads, Document Summarization, JSON Extraction, and RAG Chat with highlighted sources.
*   **Accuracy Evaluation Suite**: Includes an `evaluate.py` script that tests the RAG pipeline against ground truth data, revealing the engine operates at 83% accuracy for recipes and 62% for medical texts.

## 🛠️ Technology Stack
*   **LLM Provider**: Groq API (`llama-3.1-8b-instant`)
*   **Vector Database**: ChromaDB (Local Persistent)
*   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **PDF Parsing**: PyPDF2
*   **Image Parsing (OCR)**: EasyOCR (PyTorch)
*   **Frontend UI**: Streamlit

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.10+ installed and your `GROQ_API_KEY` defined in `.env.txt`.

### 2. Install Dependencies
```powershell
pip install groq chromadb sentence-transformers PyPDF2 easyocr opencv-python-headless streamlit python-dotenv
```

### 3. Start the Web App
Simply run the following command in your terminal:
```powershell
python -m streamlit run app.py
```

### 4. Running the Accuracy Evaluator (Optional)
To test how well the agent retrieves and answers questions against a ground truth dataset:
```powershell
python evaluate.py
```

## 🏗️ Architecture
See `expansion_architecture.md` for a full breakdown of how this Python backend scales into a React Native Mobile "Friend" App.
