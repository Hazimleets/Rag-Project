import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document

# ------------------- Utilities -------------------

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
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
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

def clean_answer(answer, question, context):
    answer = " ".join(answer.split())
    return answer if answer.strip() else "No clear answer found in document."

# ------------------- RAG Class -------------------

class DocumentRAG:
    def __init__(self, chunks, embeddings, embedder, qa_pipeline):
        self.chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        self.embeddings = embeddings
        self.embedder = embedder
        self.qa_pipeline = qa_pipeline

        if not self.chunks:
            raise ValueError("No valid text chunks found.")

        self.vectorizer = TfidfVectorizer()
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        except ValueError as e:
            raise ValueError(f"TF-IDF error: {str(e)}")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def query(self, question, top_k=3, return_chunks=False):
        question_tfidf = self.vectorizer.transform([question])
        tfidf_scores = (self.tfidf_matrix * question_tfidf.T).toarray().flatten()
        top_k_tfidf = np.argsort(tfidf_scores)[-min(top_k, len(self.chunks)):][::-1]
        candidate_chunks = [self.chunks[i] for i in top_k_tfidf]
        candidate_indices = top_k_tfidf.tolist()

        if not candidate_chunks:
            return "No relevant chunks found.", [], []

        q_embedding = self.embedder.encode([question])
        candidate_embeddings = self.embeddings[candidate_indices]

        temp_index = faiss.IndexFlatL2(candidate_embeddings.shape[1])
        temp_index.add(candidate_embeddings)
        D, I = temp_index.search(np.array(q_embedding), min(top_k, len(candidate_chunks)))

        retrieved_texts = [candidate_chunks[i] for i in I[0] if i < len(candidate_chunks)]

        context = " ".join(retrieved_texts)[:512]
        result = self.qa_pipeline(question=question, context=context)
        answer = clean_answer(result.get("answer", ""), question, context)

        return (answer, retrieved_texts, tfidf_scores[top_k_tfidf]) if return_chunks else answer

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="RAG QA 📄", layout="wide")
st.title("📄 Document-Based QA (Generic)")

if st.button("🧹 Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Cache cleared.")

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
        st.success("✅ Document loaded.")

        chunk_size = st.slider("🔢 Chunk Size", 100, 500, 150, 10)
        chunk_overlap = st.slider("🔁 Chunk Overlap", 0, 100, 20, 10)

        with st.spinner("📚 Splitting & embedding..."):
            chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            if not chunks:
                st.error("❌ No valid text chunks found.")
                st.stop()

            embedder = load_embedder()
            embeddings = embed_chunks(chunks, _embedder=embedder)
            qa_pipeline = load_qa_pipeline()

            @st.cache_resource(show_spinner=False, hash_funcs={SentenceTransformer: lambda _: None, pipeline: lambda _: None})
            def get_rag(_chunks, _embeddings, _embedder, _qa_pipeline):
                return DocumentRAG(_chunks, _embeddings, _embedder, _qa_pipeline)

            try:
                rag = get_rag(chunks, embeddings, embedder, qa_pipeline)
                st.success(f"🔗 Indexed {len(chunks)} chunks.")
            except ValueError as ve:
                st.error(f"❌ Error building index: {str(ve)}")
                st.stop()

        st.subheader("💬 Ask a question about the document")
        question = st.text_input("Enter your question:")
        top_k = st.slider("🔍 Top K Chunks", 1, 10, 3)
        debug_mode = st.checkbox("🛠️ Debug Mode")

        if st.button("Get Answer"):
            if question:
                with st.spinner("🤖 Thinking..."):
                    answer, used_chunks, tfidf_scores = rag.query(question, top_k=top_k, return_chunks=True)

                st.success("📝 Answer:")
                st.markdown(f"**{answer}**")

                with st.expander("🔎 Retrieved Chunks"):
                    for i, chunk in enumerate(used_chunks, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text_area(f"Chunk {i}", chunk, height=150)

                if debug_mode:
                    st.subheader("🛠️ Debug Info")
                    st.write(f"TF-IDF Scores: {tfidf_scores}")
            else:
                st.warning("⚠️ Please enter a question.")
