import sys
try:
    sys.modules["sqlite3"] = __import__("pysqlite3")
except ImportError:
    # Use built-in sqlite3 if pysqlite3 is not installed
    pass
import streamlit as st
from rag_utils import process_uploaded_file, query_chroma, get_chroma_collection, STRUCTURED_DATA, hybrid_answer
import os
import re
import openai

# --- LLM Model Choices ---
SMALL_MODEL = "distilgpt2"
LARGE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def set_llm_model(model_name):
    import local_llm_utils
    from transformers import AutoTokenizer, AutoModelForCausalLM
    local_llm_utils.DEFAULT_MODEL = model_name
    local_llm_utils._llm_pipeline = None
    try:
        with st.spinner(f"Downloading model '{model_name}', please wait (one-time download)..."):
            AutoTokenizer.from_pretrained(model_name)
            AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Model download failed: {e}")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="💼",
    layout="wide"
)

# --- Modern, Professional CSS ---
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #2d3e50 0%, #4f8ef7 100%);
        color: white;
        padding: 32px 0 18px 0;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 6px 24px rgba(50,50,50,0.10);
        margin-bottom: 18px;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    .chat-container {
        background: #fff;
        border-radius: 18px;
        padding: 24px;
        margin: 20px 0 10px 0;
        box-shadow: 0 8px 32px rgba(80,80,120,0.08);
        max-height: 60vh;
        overflow-y: auto;
    }
    .message {
        padding: 14px 20px;
        margin: 12px 0;
        border-radius: 14px;
        max-width: 80%;
        font-size: 1.08rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(80,80,120,0.04);
    }
    .user-message {
        background: linear-gradient(90deg, #4f8ef7 30%, #2d3e50 100%);
        color: white;
        margin-left: auto;
        text-align: right;
        border-bottom-right-radius: 4px;
        border-top-right-radius: 4px;
    }
    .bot-message {
        background: linear-gradient(90deg, #f6d365 0%, #fda085 100%);
        color: #2d3e50;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border-top-left-radius: 4px;
    }
    .context-chunk {
        background: #f4f8ff;
        color: #2d3e50;
        border-left: 4px solid #4f8ef7;
        padding: 10px 14px;
        margin: 8px 0 0 0;
        font-size: 0.98rem;
        border-radius: 8px;
    }
    .highlight {
        background: #ffe066;
        font-weight: 600;
        border-radius: 4px;
        padding: 2px 4px;
    }
    .file-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 0.97rem;
    }
    .file-table th, .file-table td {
        border: 1px solid #e0e7ef;
        padding: 8px 12px;
        text-align: left;
    }
    .file-table th {
        background: #f4f8ff;
        color: #2d3e50;
    }
    .upload-section {
        background: rgba(255, 255, 255, 0.13);
        border-radius: 15px;
        padding: 22px;
        margin: 18px 0;
        border: 2px dashed #e0e7ef;
    }
    .status-success {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .status-error {
        background: linear-gradient(90deg, #f44336 0%, #da190b 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 8px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Modern Header ---
st.markdown("""
<div class='main-header'>
    <h1>RAG Assistant</h1>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 **Upload Documents**")
    uploaded_files = st.file_uploader(
        "Upload your files",
        type=["pdf", "txt", "md", "csv", "json", "xlsx", "docx"],
        accept_multiple_files=True,
        help="Upload PDFs, CSV files, or any document format"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Processing status and file preview table
    if 'uploaded_file_status' not in st.session_state:
        st.session_state['uploaded_file_status'] = {}
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state['uploaded_file_status']:
                ext = file.name.split('.')[-1].lower()
                if ext in ['pdf', 'txt', 'md', 'csv', 'json', 'xlsx', 'docx']:
                    with st.spinner(f"Processing {file.name}..."):
                        try:
                            num_chunks = process_uploaded_file(file, ext)
                            st.session_state['uploaded_file_status'][file.name] = f"✅ Processed ({num_chunks} chunks)"
                        except Exception as e:
                            st.session_state['uploaded_file_status'][file.name] = f"❌ Error: {str(e)}"
                else:
                    st.session_state['uploaded_file_status'][file.name] = "❌ Unsupported format"
        st.markdown("**Uploaded Files:**")
        st.markdown("<table class='file-table'><tr><th>File</th><th>Status</th></tr>" + ''.join([
            f"<tr><td>{fname}</td><td>{status}</td></tr>" for fname, status in st.session_state['uploaded_file_status'].items()
        ]) + "</table>", unsafe_allow_html=True)
    else:
        st.info("Upload files to start analyzing")

    st.markdown("---")
    st.markdown("### 🔑 **OpenAI API Key**")
    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = ''
    if 'use_openai' not in st.session_state:
        st.session_state['use_openai'] = False
    st.session_state['openai_api_key'] = st.text_input("Enter your OpenAI API key", type="password", value=st.session_state['openai_api_key'])
    st.session_state['use_openai'] = st.checkbox("Use OpenAI for answers (requires API key)", value=False, disabled=(not st.session_state['openai_api_key']))
    st.session_state['openai_model'] = st.selectbox(
        "OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0,
        disabled=not st.session_state['use_openai']
    )
    if st.session_state['use_openai'] and st.session_state['openai_model'] == "gpt-4":
        st.info("Make sure your API key has GPT-4 access. If not, use gpt-3.5-turbo.")

    st.markdown("---")
    st.markdown("### 🔑 **Fireworks API Key**")
    if 'fireworks_api_key' not in st.session_state:
        st.session_state['fireworks_api_key'] = ''
    st.session_state['fireworks_api_key'] = st.text_input("Enter your Fireworks API key", type="password", value=st.session_state['fireworks_api_key'])
    # Set the environment variable at runtime for backend use
    if st.session_state['fireworks_api_key']:
        os.environ['FIREWORKS_API_KEY'] = st.session_state['fireworks_api_key']

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state['chat_history'] = []
            st.rerun()
    with col2:
        if st.button("🗂️ Clear Data", use_container_width=True):
            st.session_state['uploaded_file_status'] = {}
            st.session_state['chat_history'] = []
            st.session_state['last_question'] = None
            get_chroma_collection(force_reset=True)
            st.rerun()

# --- Main Area ---
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state['chat_history']:
        st.markdown(f'<div class="message user-message"><strong>You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message bot-message"><strong>AI:</strong> {chat["bot"]}</div>', unsafe_allow_html=True)
        if 'contexts' in chat:
            for idx, ctx in enumerate(chat['contexts']):
                st.markdown(f'<div class="context-chunk" style="display: none;"><em>Context {idx+1}:</em><br>{ctx}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Show summary of uploaded files and extracted tables
if st.session_state.get('uploaded_file_status'):
    st.markdown('<div style="margin-top: 24px;">', unsafe_allow_html=True)
    st.markdown('#### Uploaded File Summary')
    for fname, status in st.session_state['uploaded_file_status'].items():
        st.write(f"**{fname}**: {status}")
        if fname in STRUCTURED_DATA:
            for i, df in enumerate(STRUCTURED_DATA[fname]):
                if df is not None and not df.empty:
                    st.markdown(f"<details><summary>Table {i+1} Preview</summary>", unsafe_allow_html=True)
                    st.dataframe(df.head(10))
                    st.markdown("</details>", unsafe_allow_html=True)

# --- Input and Answer Logic ---
# --- Input and Answer Logic with Streamlit Form ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask questions about your documents:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state['last_question'] = user_input
    with st.spinner("🔍 Searching for answers..."):
        context_chunks = query_chroma(user_input, top_k=3)
        matched_sentences = []
        highlighted_contexts = []
        for chunk in context_chunks:
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            best_sent = max(sentences, key=lambda s: sum(1 for w in user_input.lower().split() if w in s.lower()))
            matched_sentences.append(best_sent.strip())
            highlighted = chunk.replace(best_sent, f'<span class="highlight">{best_sent}</span>')
            highlighted_contexts.append(highlighted)
        extracted_answer = ' '.join(matched_sentences) if matched_sentences else "No relevant information found in the documents."

        openai_answer = None
        if st.session_state.get('use_openai') and st.session_state.get('openai_api_key'):
            try:
                openai.api_key = st.session_state['openai_api_key']
                prompt = (
                    "You are an expert assistant. Answer the following user question using ONLY the provided context. "
                    "If the answer is not present in the context, say 'Not found in the provided documents.'\n"
                    f"Context:\n{chr(10).join([re.sub('<.*?>', '', chunk) for chunk in context_chunks])}\n"
                    f"Question: {user_input}\nAnswer: "
                )
                response = openai.chat.completions.create(
                    model=st.session_state['openai_model'],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.0
                )
                openai_answer = response.choices[0].message.content.strip()
            except Exception as e:
                openai_answer = f"[OpenAI API Error] {str(e)}"

        if openai_answer:
            answer = f"**OpenAI Answer:** {openai_answer}"
        else:
            llm_answer = None
            try:
                llm_raw = hybrid_answer(user_input)
                rep_words = ["code of conduct", "policy", "not a", "emergency staff"]
                if any(word in llm_raw.lower() for word in rep_words) and len(llm_raw.split()) < 20:
                    llm_answer = None
                else:
                    llm_answer = llm_raw
            except Exception:
                llm_answer = None
            # Show LLM answer if strong, else best sentence from Context 1
            import re
            try:
                if llm_answer and len(llm_answer.strip()) > 10:
                    answer = f"**Answer:** {llm_answer.strip()}"
                else:
                    best_snippet = None
                    if context_chunks and len(context_chunks) > 0:
                        qwords = set(re.findall(r'\w+', user_input.lower()))
                        best_score = 0
                        for sent in re.split(r'(?<=[.!?])\s+', context_chunks[0]):
                            sent_words = set(re.findall(r'\w+', sent.lower()))
                            score = len(qwords & sent_words)
                            if score > best_score and len(sent) > 10:
                                best_snippet = sent
                                best_score = score
                    if best_snippet:
                        answer = f"**Answer:** {best_snippet.strip()}"
                    else:
                        answer = "**Answer:** No relevant answer found."
            except Exception as e:
                answer = f"**Error:** An unexpected error occurred during answer selection: {str(e)}"

        st.session_state['chat_history'].append({
            "user": user_input,
            "bot": answer,
            "contexts": highlighted_contexts
        })
        st.rerun()
