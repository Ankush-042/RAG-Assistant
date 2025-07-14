import pdfplumber
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
import re
import pandas as pd
import json
from docx import Document

def extract_text_from_pdf(file) -> (str, list):
    """Improved PDF extraction: splits text into paragraphs for better chunking, with debug logging."""
    try:
        import logging
        with pdfplumber.open(file) as pdf:
            all_text = []
            all_tables = []
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    # Split by lines for more granularity
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    all_text.extend(lines)
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0]) if table and len(table) > 1 else None
                    if df is not None:
                        all_tables.append(df)
            joined_text = '\n\n'.join(all_text)
            print(f"[PDF Extraction] Extracted {len(all_text)} paragraphs/lines from PDF.")
            return joined_text, all_tables
    except Exception as e:
        print(f"[PDF Extraction] Error: {str(e)}")
        return f"Error reading PDF: {str(e)}", []

def extract_text_from_txt(file) -> (str, list):
    return file.read().decode("utf-8"), []

def extract_text_from_md(file) -> (str, list):
    return file.read().decode("utf-8"), []

def extract_text_from_csv(file) -> (str, list):
    """Extract text and DataFrame from CSV files. Each row is a chunk."""
    try:
        df = pd.read_csv(file)
        # Each row as a chunk
        row_chunks = [', '.join([f'{col}: {row[col]}' for col in df.columns]) for _, row in df.iterrows()]
        return '\n'.join(row_chunks), [df]
    except Exception as e:
        return f"Error reading CSV: {str(e)}", []

def extract_text_from_json(file) -> (str, list):
    """Extract text and DataFrame from JSON files (if possible)"""
    try:
        data = json.load(file)
        # Try to load as DataFrame if tabular
        df = None
        if isinstance(data, list) and all(isinstance(row, dict) for row in data):
            df = pd.DataFrame(data)
        return json.dumps(data, indent=2), [df] if df is not None else []
    except Exception as e:
        return f"Error reading JSON: {str(e)}", []

def extract_text_from_excel(file) -> (str, list):
    """Extract text and DataFrames from Excel files (all sheets). Each row is a chunk."""
    try:
        xls = pd.ExcelFile(file)
        dfs = [xls.parse(sheet) for sheet in xls.sheet_names]
        row_chunks = []
        for df in dfs:
            row_chunks.extend([', '.join([f'{col}: {row[col]}' for col in df.columns]) for _, row in df.iterrows()])
        text = '\n'.join(row_chunks)
        return text, dfs
    except Exception as e:
        return f"Error reading Excel: {str(e)}", []

def extract_text_from_docx(file) -> (str, list):
    """Extract text from Word documents"""
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text, []
    except Exception as e:
        return f"Error reading Word document: {str(e)}", []

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 2) -> List[str]:
    """Fine-grained, overlapping sentence chunking (2-3 sentences per chunk)."""
    import re
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = ' '.join(sentences[i:i+3])
        if len(chunk) > 50:
            chunks.append(chunk)
        i += 3 - overlap  # Slide window with overlap
    return chunks

# Simple embedding model loading
_embedding_model = None
# Global dict to store structured DataFrames per file
STRUCTURED_DATA = {}

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return None
    return _embedding_model

def embed_chunks(chunks: list) -> list:
    """Simple embedding generation"""
    model = get_embedding_model()
    if model is None:
        return []
    try:
        return model.encode(chunks).tolist()
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def get_chroma_collection(persist_dir="chroma_db"):
    """Get ChromaDB collection"""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection("kb_chunks")
        return collection
    except Exception as e:
        print(f"Error getting ChromaDB collection: {e}")
        return None

def add_chunks_to_chroma(chunks: list, embeddings: list, file_name: str, metadatas: list = None):
    """Add chunks to ChromaDB"""
    collection = get_chroma_collection()
    if collection is None:
        return
    
    try:
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        
        # Create metadata for each chunk
        if metadatas is None:
            metadatas = [{"source": file_name, "chunk_index": i} for i in range(len(chunks))]
        
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    except Exception as e:
        print(f"Error adding chunks to ChromaDB: {e}")

def query_chroma(query: str, top_k: int = 10):
    """Simple query function"""
    collection = get_chroma_collection()
    if collection is None:
        return []
    
    model = get_embedding_model()
    if model is None:
        return []
    
    try:
        query_emb = model.encode([query]).tolist()[0]
        
        results = collection.query(
            query_embeddings=[query_emb], 
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Filter out results that are too dissimilar
        filtered_docs = []
        for doc, dist in zip(documents, distances):
            if dist < 1.5:  # Relaxed threshold
                filtered_docs.append(doc)
        
        return filtered_docs if filtered_docs else documents
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def process_uploaded_file(file, filetype: str):
    """Process uploaded file: extract text, tables, embed, and store"""
    file_name = getattr(file, 'name', 'uploaded')
    if hasattr(file, 'seek'):
        file.seek(0)
    
    try:
        # Extract text and tables based on file type
        if filetype == 'pdf':
            text, tables = extract_text_from_pdf(file)
        elif filetype == 'txt':
            text, tables = extract_text_from_txt(file)
        elif filetype == 'md':
            text, tables = extract_text_from_md(file)
        elif filetype == 'csv':
            text, tables = extract_text_from_csv(file)
        elif filetype == 'json':
            text, tables = extract_text_from_json(file)
        elif filetype == 'xlsx':
            text, tables = extract_text_from_excel(file)
        elif filetype == 'docx':
            text, tables = extract_text_from_docx(file)
        else:
            return 0
        # Store tables for direct lookup
        if tables:
            STRUCTURED_DATA[file_name] = tables
        # Chunk the text
        chunks = chunk_text(text)
        if not chunks:
            return 0
        # Generate embeddings
        embeddings = embed_chunks(chunks)
        if not embeddings:
            return 0
        # Add to ChromaDB
        add_chunks_to_chroma(chunks, embeddings, file_name)
        return len(chunks)
    except Exception as e:
        raise Exception(f"Error processing {file_name}: {str(e)}")

# Hybrid answer: search tables first, then RAG fallback
from local_llm_utils import synthesize_answer_with_llm
try:
    from fireworks_utils import synthesize_answer_with_fireworks
except ImportError:
    synthesize_answer_with_fireworks = None

from analysis_utils import analyze_structured_data

def hybrid_answer(query: str, file_name_hint: str = None):
    import re
    # Try to match queries like 'score of X', 'did X pass', 'marks of X in Y', etc.
    tables_to_search = []
    if file_name_hint and file_name_hint in STRUCTURED_DATA:
        tables_to_search = STRUCTURED_DATA[file_name_hint]
    else:
        for tlist in STRUCTURED_DATA.values():
            tables_to_search.extend(tlist)
    # Analytical/statistical query check
    if tables_to_search:
        analysis_result = analyze_structured_data(query, tables_to_search)
        if analysis_result:
            return analysis_result
    name_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+|[A-Za-z]+)', query)
    name = name_match.group(1) if name_match else None
    subj_match = re.search(r'(in|of)\s+([A-Za-z ]+)', query)
    subject = subj_match.group(2).strip() if subj_match else None
    for df in tables_to_search:
        if df is not None:
            for idx, row in df.iterrows():
                row_str = ' '.join(str(x) for x in row.values)
                if name and name.lower() in row_str.lower():
                    if subject:
                        for col in df.columns:
                            if subject.lower() in col.lower():
                                return f"**{name}** - **{col}**: {row[col]}"
                    # Return only the row as a formatted table
                    return ' | '.join([f"**{col}**: {row[col]}" for col in df.columns])
    # fallback to RAG
    # Advanced semantic re-ranking: get top 8 candidates, then re-rank with cross-encoder
    from cross_encoder_utils import rerank_chunks
    # Hybrid: embedding + BM25 sparse retrieval with caching
    if not hasattr(hybrid_answer, 'retrieval_cache'):
        hybrid_answer.retrieval_cache = {}
    cache_key = (query,)
    if cache_key in hybrid_answer.retrieval_cache:
        dense_chunks = hybrid_answer.retrieval_cache[cache_key]
    else:
        dense_chunks = query_chroma(query, top_k=5)
        hybrid_answer.retrieval_cache[cache_key] = dense_chunks
    # BM25 sparse retrieval
    from bm25_utils import bm25_retrieve
    # Use all chunks in ChromaDB for BM25 (simulate by reusing dense_chunks for now; can be replaced with all chunks if needed)
    bm25_chunks_scores = bm25_retrieve(query, dense_chunks, k=5)
    bm25_chunks = [c for c, s in bm25_chunks_scores]
    # Merge and deduplicate
    all_candidates = list(dict.fromkeys([c for c in dense_chunks + bm25_chunks if c and len(c.strip()) > 40]))
    from cross_encoder_utils import rerank_chunks
    context_chunks = rerank_chunks(query, all_candidates, top_k=2)
    if context_chunks:
        context = '\n'.join(context_chunks)
        # Concise, direct prompt for LLM
        # Improved prompt for LLM: extract or summarize answer if present, else say Not found
        improved_prompt = f"""
        You are an HR assistant. Use ONLY the following context to answer the question. If the answer is present, quote or summarize it concisely. If not, reply 'Not found.'
        Context:
        {context}
        Question: {query}
        """
        try:
            answer = synthesize_answer_with_llm(query, context, prompt=improved_prompt) if 'prompt' in synthesize_answer_with_llm.__code__.co_varnames else synthesize_answer_with_llm(query, context)
        except Exception as e:
            print(f"[LLM Error] {e}")
            answer = None
        # Loosened answer validation: allow short, context-based answers
        def is_bad_answer(ans, ctxs):
            if not ans: return True
            ans_norm = ans.strip().lower()
            # Block only if answer is a near-exact context dump or totally irrelevant
            if len(ans_norm) > 300 and any(ans_norm in c.lower() or c.lower() in ans_norm for c in ctxs):
                return True
            if ans_norm in ["not found", "not found in the provided documents.", "no relevant information found."]:
                return True
            return False
        if (is_bad_answer(answer, context_chunks) or 'Not found' in str(answer)) and synthesize_answer_with_fireworks:
            try:
                answer_fw = synthesize_answer_with_fireworks(query, context)
                if answer_fw and not is_bad_answer(answer_fw, context_chunks) and 'Error' not in str(answer_fw):
                    return answer_fw
            except Exception as e:
                print(f"[Fireworks Error] {e}")
                pass
        if not is_bad_answer(answer, context_chunks):
            return answer
        # Fallback: extract most relevant sentence(s) from context
        import re
        qwords = set(re.findall(r'\w+', query.lower()))
        best_sent = ''
        best_score = 0
        for chunk in context_chunks:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                sent_words = set(re.findall(r'\w+', sent.lower()))
                score = len(qwords & sent_words)
                if score > best_score and len(sent) > 20:
                    best_sent = sent
                    best_score = score
        if best_score > 0:
            return best_sent.strip()
        else:
            return "Not found in the provided documents."
    return "No relevant information found."