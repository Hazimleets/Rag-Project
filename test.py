import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------- Utilities -------------------

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"⚠️ Error processing PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"⚠️ Error processing DOCX: {str(e)}")
        return ""

def chunk_text(text, chunk_size=150, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=True
    )
    return splitter.split_text(text)

@st.cache_resource(show_spinner=False)
def load_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def embed_chunks(chunks, _embedder):
    embeddings = _embedder.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

def clean_answer(answer, question, context):
    answer = " ".join(answer.split())
    if "technical skills" in question.lower() and "●" in context:
        skills = [line.strip("● ").strip() for line in context.split("\n") if line.startswith("●")]
        return "\n".join(skills) if skills else answer
    return answer

# ------------------- RAG Class -------------------

class DocumentRAG:
    def __init__(self, chunks, embeddings, embedder, qa_pipeline):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.qa_pipeline = qa_pipeline
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def query(self, question, top_k=3, return_chunks=False):
        # Step 1: Keyword-based filtering with TF-IDF
        question_tfidf = self.vectorizer.transform([question])
        tfidf_scores = (self.tfidf_matrix * question_tfidf.T).toarray().flatten()
        top_k_tfidf = np.argsort(tfidf_scores)[-min(top_k, len(self.chunks)):][::-1]
        candidate_chunks = [self.chunks[i] for i in top_k_tfidf]
        candidate_indices = top_k_tfidf.tolist()

        if not candidate_chunks:
            return "No relevant chunks found.", [], tfidf_scores[top_k_tfidf] if return_chunks else "No relevant chunks found."

        # Step 2: Embedding-based retrieval on candidates
        q_embedding = self.embedder.encode([question])
        candidate_embeddings = self.embeddings[candidate_indices]

        # Create a temporary FAISS index for candidates
        dim = candidate_embeddings.shape[1]
        temp_index = faiss.IndexFlatL2(dim)
        temp_index.add(candidate_embeddings)
        D, I = temp_index.search(np.array(q_embedding), min(top_k, len(candidate_chunks)))

        # Map back to original chunk indices
        retrieved_texts = [candidate_chunks[i] for i in I[0] if i < len(candidate_chunks)]

        # Step 3: QA pipeline
        context = " ".join(retrieved_texts)[:512]
        result = self.qa_pipeline(question=question, context=context)
        answer = clean_answer(result["answer"], question, context)

        if return_chunks:
            return answer, retrieved_texts, tfidf_scores[top_k_tfidf]
        return answer

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="RAG QA 📄", layout="wide")
st.title("📄 Document-Based QA with RAG")

if st.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Cache cleared. Please reload your document.")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raw_text = extract_text_from_docx(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {str(e)}")
        raw_text = ""

    if raw_text:
        raw_text = " ".join(raw_text.split())
        st.success("✅ Document loaded successfully.")

        chunk_size = st.slider("🔢 Chunk Size", min_value=100, max_value=500, value=150, step=10)
        chunk_overlap = st.slider("🔁 Chunk Overlap", min_value=0, max_value=100, value=20, step=10)

        with st.spinner("📚 Splitting and embedding text..."):
            chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            embedder = load_embedder()
            embeddings = embed_chunks(chunks, _embedder=embedder)
            qa_pipeline = load_qa_pipeline()

            @st.cache_resource(show_spinner=False, hash_funcs={SentenceTransformer: lambda _: None, pipeline: lambda _: None})
            def get_rag(_chunks, _embeddings, _embedder, _qa_pipeline):
                return DocumentRAG(_chunks, _embeddings, _embedder, _qa_pipeline)

            rag = get_rag(chunks, embeddings, embedder, qa_pipeline)
            st.success(f"🔗 Indexed {len(chunks)} text chunks.")

        st.subheader("💬 Ask a question about the document")
        question = st.text_input("Enter your question:")
        top_k = st.slider("🔍 Top K Chunks to Retrieve", min_value=1, max_value=10, value=3)
        debug_mode = st.checkbox("🛠️ Debug Mode")

        if st.button("Get Answer"):
            if question:
                if len(question.split()) < 3:
                    st.warning("⚠️ Please ask a more specific question.")
                else:
                    with st.spinner("🤖 Generating answer..."):
                        answer, used_chunks, tfidf_scores = rag.query(question, top_k=top_k, return_chunks=True)
                    st.success("📝 Answer:")
                    if "technical skills" in question.lower():
                        st.table({"Skills": answer.split("\n")})
                    else:
                        st.markdown(f"**{answer}**")

                    with st.expander("🔎 View Retrieved Chunks"):
                        for i, chunk in enumerate(used_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text_area(f"Chunk {i}", chunk, height=150)

                    if debug_mode:
                        st.subheader("🛠️ Debug Information")
                        st.write(f"TF-IDF Scores for Top Chunks: {tfidf_scores}")
            else:
                st.warning("⚠️ Please enter a question.")