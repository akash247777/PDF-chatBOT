import streamlit as st
import fitz, io, os, requests, time
import pytesseract                       # <- NEW
from PIL import Image                    # <- NEW
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# ---------- TESSERACT PATH ----------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

HF_API_KEY = "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"

os.environ["HF_TOKEN"] = HF_API_KEY  # Ensure your token is set
client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_TOKEN"],
)

MODEL_CHAT = "moonshotai/Kimi-K2-Instruct"
HEADERS_JSON = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
encoder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------- HELPERS ----------
def ocr_bytes(img_bytes: bytes) -> str:
    """OCR an image via Tesseract."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img, lang="eng").strip()
    except Exception as e:
        return f"[OCR failed: {e}]"

def ingest_files(uploaded_files):
    docs = []
    for file in uploaded_files:
        file_bytes = file.read()
        text = ""

        if file.type == "application/pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text()
                for _, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    img_data = pix.tobytes("png")
                    text += "\n" + ocr_bytes(img_data)
        else:  # image
            text = ocr_bytes(file_bytes)

        docs.append(Document(page_content=text, metadata={"source": file.name}))

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(docs)
    st.session_state["vectorstore"] = FAISS.from_documents(chunks, encoder)
    return bool(chunks)

def query_kimi(messages, context):
    """Call HF endpoint for Kimi-K2-Instruct using InferenceClient."""
    system = (
        "Answer strictly from the context below in detail. "
        "If not found, reply exactly: \"I couldn’t find that information in the provided documents.\"\n\n"
        f"Context:\n{context}"
    )
    full_messages = [{"role": "system", "content": system}] + messages
    try:
        completion = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=full_messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ {MODEL_CHAT} is currently unavailable. Error: {e}"

# ---------- UI ----------
st.set_page_config(page_title="Kimi-K2 + Tesseract RAG")
st.title("📄 Chat with your documents (text + Tesseract OCR)")

uploaded = st.file_uploader("Upload PDF(s) or image(s)",
                            type=["pdf", "png", "jpg", "jpeg"],
                            accept_multiple_files=True)
if uploaded and st.button("Process files"):
    with st.spinner("Extracting text + OCR + indexing…"):
        ingest_files(uploaded)
    st.success("✅ Ready! Ask questions below.")

if "vectorstore" not in st.session_state:
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask something…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    docs = st.session_state.vectorstore.similarity_search(user_q, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    answer = query_kimi(st.session_state.messages, context)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)