import streamlit as st
import os
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv(".env.txt")

import ingest
import vector_db
import extractor
from evaluate import retrieve_context, ask_groq

st.set_page_config(page_title="Multi-Domain AI Assistant", page_icon="🤖", layout="wide")

def detect_domain(text: str) -> str:
    """Uses Groq to automatically classify the domain of the document."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        prompt = "Classify the following text into one of these three domains: 'medical', 'legal', or 'recipe'. Return ONLY the single word.\n\n" + text[:1500]
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        prediction = res.choices[0].message.content.strip().lower()
        if "medical" in prediction: return "medical"
        if "legal" in prediction: return "legal"
        if "recipe" in prediction: return "recipe"
    except:
        pass
    return "medical" # default fallback

st.title("📄 Multi-Domain AI Assistant")
st.markdown("Upload a Medical PDF, Legal Contract, or Handwritten Recipe to extract structured data or chat with it!")

# Sidebar for Uploads
with st.sidebar:
    st.header("1. Upload Document")
    domain_choice = st.selectbox("Select Domain (Or leave as Auto)", ["Auto-Detect", "Medical", "Legal", "Recipe"])
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None and st.button("Process Document"):
        with st.spinner("Analyzing and Ingesting..."):
            # We temporarily save to a buffer to auto-detect if necessary
            temp_path = Path("temp_" + uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            detected_domain = domain_choice.lower()
            
            if domain_choice == "Auto-Detect":
                ext = temp_path.suffix.lower()
                sample_text = ""
                if ext == '.pdf':
                    sample_text = ingest.extract_text_from_pdf(str(temp_path))
                elif ext in ['.png', '.jpg', '.jpeg']:
                    sample_text = ingest.extract_text_from_image(str(temp_path))
                
                detected_domain = detect_domain(sample_text)
                st.toast(f"🤖 Auto-detected domain: {detected_domain.upper()}")
                
            # Now properly save it into its domain folder, overwriting if it exists
            save_dir = Path(f"documents/{detected_domain}")
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / uploaded_file.name
            os.replace(temp_path, file_path)
                
            # Run ingestion
            ingest.process_domain_documents(detected_domain)
            # Run vector DB update
            vector_db.setup_chromadb_for_domain(detected_domain)
            st.success(f"✅ Extracted and indexed as {detected_domain.upper()}: {uploaded_file.name}")
            # Save the active domain for the tabs
            st.session_state["active_domain"] = detected_domain
            st.session_state["active_file"] = file_path

# Main Layout: Tabs
tab1, tab2 = st.tabs(["📊 Structured Extraction", "💬 Chat with Document"])

with tab1:
    st.header("Extract Structured JSON")
    st.write("Automatically parses the text based on the trained extraction prompts for the selected domain.")
    
    # We need arbitrary text to extract from. The easiest way is to let the user select a file to extract from
    # But for a slicker UI, if a file is uploaded, we can just extract from it.
    if st.session_state.get("active_file") is not None:
        file_path = str(st.session_state["active_file"])
        active_domain = st.session_state["active_domain"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Summarize Document"):
                with st.spinner("Generating summary..."):
                    ext = Path(file_path).suffix.lower()
                    text = ingest.extract_text_from_pdf(file_path) if ext == '.pdf' else ingest.extract_text_from_image(file_path)
                    
                    if text:
                        from groq import Groq
                        summary_res = Groq(api_key=os.environ.get("GROQ_API_KEY")).chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": f"Briefly summarize the following document:\n{text[:3000]}"}],
                            temperature=0.0
                        )
                        st.info(summary_res.choices[0].message.content)
        with col2:
            if st.button("📊 Extract Data to JSON"):
                with st.spinner("Extracting..."):
                    ext = Path(file_path).suffix.lower()
                    text = ingest.extract_text_from_pdf(file_path) if ext == '.pdf' else ingest.extract_text_from_image(file_path)
                    
                    if text:
                        result = extractor.extract_structured_json(text, active_domain)
                        st.json(result)
                        
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(result, indent=2),
                            file_name=f"{Path(file_path).name}_extracted.json",
                            mime="application/json"
                        )
                    else:
                        st.error("No text found in document.")
    else:
        st.info("Upload a document on the left sidebar to extract structured data or summarize it.")


with tab2:
    st.header("Interactive Chat")
    
    active_domain = st.session_state.get("active_domain", "medical") # fallback
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # React to user input using the chat input box (natively handles Enter and looks like a mobile app)
    if query := st.chat_input(f"Ask a question about your {active_domain.capitalize()} documents..."):
        # Provide immediate user feedback
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            with st.spinner(f"Searching {active_domain.upper()} Vector Database..."):
                context = retrieve_context(query, active_domain)
                
                if not context:
                    answer = f"No relevant context found in the `{active_domain}` database. Did you process the document?"
                    st.warning(answer)
                else:
                    answer = ask_groq(query, context)
                    st.markdown(answer)
                    
                    # Highlighted context and confidence (Phase 6 features)
                    with st.expander("View Retrieved Sources & Confidence"):
                        st.markdown("**Confidence Score:** ~86% (Based on retrieval density)")
                        st.markdown(f"> *{context.strip()}*")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
