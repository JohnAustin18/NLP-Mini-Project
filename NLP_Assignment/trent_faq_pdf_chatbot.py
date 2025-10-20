import os
import json
import numpy as np
import requests
import streamlit as st

# Import sklearn with error handling
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.error("Please ensure scikit-learn is installed. Check your requirements.txt file.")
    st.stop()

try:
    from PyPDF2 import PdfReader
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.error("Please ensure PyPDF2 is installed. Check your requirements.txt file.")
    st.stop()

# --- Configuration ---
# Hardcoded default key (can be overridden via sidebar)
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-3.5-turbo"
# Use absolute path resolution to ensure it works when deployed
FAQ_PDF_PATH = os.path.join(os.path.dirname(__file__), "FAQs_questions.pdf")
MAX_SNIPPETS = 5
MAX_SNIPPET_CHARS = 1000

# --- Utilities ---
def get_api_key() -> str:
    # Priority: hardcoded variable, Streamlit secrets, environment
    key = OPENROUTER_API_KEY
    if key and key != "sk-or-v1-REPLACE_ME":
        return key
    key = st.secrets.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
    return key

def validate_openrouter_key(api_key: str) -> tuple[bool, str]:
    """Validate API key by calling OpenRouter models endpoint."""
    if not api_key:
        return False, "API key is empty."
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://localhost",
                "X-Title": "Trent FAQ Chatbot",
            },
            timeout=20,
        )
        if resp.status_code == 200:
            return True, ""
        else:
            try:
                data = resp.json()
                msg = data.get("error", {}).get("message", resp.text)
            except Exception:
                msg = resp.text
            return False, f"{resp.status_code} {msg}"
    except Exception as e:
        return False, str(e)

def _read_faq_pdf(pdf_path: str) -> list:
    """Return list of dicts [{'page_number': int, 'text': str, 'source': 'faq_pdf'}] from the PDF only."""
    pages: list[dict] = []
    try:
        # Check if file exists first
        if not os.path.exists(pdf_path):
            st.error(f"FAQ PDF not found at '{pdf_path}'")
            st.error(f"Current working directory: {os.getcwd()}")
            st.error(f"Files in current directory: {os.listdir('.')}")
            st.error("Please ensure 'FAQs_questions.pdf' is in the same directory as the app.")
            return pages
            
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append({"page_number": i, "text": text.strip(), "source": "faq_pdf"})
        st.success(f"Successfully loaded FAQ PDF with {len(pages)} pages")
    except Exception as e:
        st.error(f"FAQ PDF couldn't be processed ({e}). The bot will not work properly.")
        st.error(f"File path attempted: {pdf_path}")
        st.error(f"File exists: {os.path.exists(pdf_path) if pdf_path else 'N/A'}")
    return pages

@st.cache_resource
def load_faq_artifacts(pdf_path: str):
    """Load FAQ PDF pages and build a TF-IDF matrix."""
    faq_pages = _read_faq_pdf(pdf_path)
    if not faq_pages:
        return [], None, None

    texts = [p.get("text", "") for p in faq_pages]

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    counts = vectorizer.fit_transform(texts)

    tfidf = TfidfTransformer(norm="l2", use_idf=True)
    tfidf_matrix = tfidf.fit_transform(counts)

    return faq_pages, tfidf_matrix, vectorizer

def retrieve_context(query: str, vectorizer: CountVectorizer, tfidf_matrix: np.ndarray, pages: list, k: int = MAX_SNIPPETS) -> list:
    query_counts = vectorizer.transform([query])
    query_vec = normalize(query_counts, norm="l2")
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    idx_sorted = np.argsort(scores)[::-1][:k]
    selected = []
    for i in idx_sorted:
        selected.append({
            "page_number": pages[i].get("page_number"),
            "text": (pages[i].get("text") or "")[:MAX_SNIPPET_CHARS],
            "score": float(scores[i]),
            "source": pages[i].get("source", "faq_pdf"),
        })
    return selected

def call_openrouter(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Trent FAQ Chatbot",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def build_prompt(question: str, snippets: list) -> tuple[str, str]:
    snippet_text = "\n\n".join([f"[FAQ Page {s['page_number']} | score {s['score']:.3f}]\n{s['text']}" for s in snippets])
    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the provided context from the Trent/Zudio FAQs PDF. "
        "Trent Limited operates brands like Westside, Zudio, Samoh, and Star. Provide accurate information based on the context provided. "
        "If the answer is not present in the context, say you couldn't find it explicitly and suggest the most relevant sections."
    )
    user_prompt = (
        f"Context:\n{snippet_text}\n\n"
        f"Question: {question}\n\n"
        "Please provide a comprehensive answer based on the context provided."
    )
    return system_prompt, user_prompt

# --- UI ---
st.set_page_config(page_title="Trent/Zudio FAQ Chatbot (PDF)", page_icon="🛍️", layout="wide")

st.markdown(
    """
<style>
    .main .block-container { padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    .stText, .stMarkdown, .element-container { width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🛍️ Trent/Zudio FAQ Chatbot (PDF-only)")
st.caption("Answers strictly from 'FAQs_questions.pdf'. Uses the same OpenRouter API as the original app.")

TOP_FAQ_QUESTIONS = [
    "What is Trent Limited and what does the company do?",
    "What brands does Trent operate?",
    "How many stores does Trent operate?",
    "What is Trent's financial performance in FY25?",
    "What is Trent's approach to sustainability?",
    "How does Trent manage its supply chain?",
]

with st.sidebar:
    st.subheader("Settings")
    st.text_input("OpenRouter API key", type="password", key="override_key")
    st.number_input("Snippets (k)", min_value=1, max_value=10, value=MAX_SNIPPETS, key="k")
    st.number_input("Max snippet chars", min_value=200, max_value=2000, value=MAX_SNIPPET_CHARS, step=100, key="max_chars")
    st.markdown("---")
    st.markdown("**Note:** This chatbot answers using only the bundled FAQs PDF.")

    if st.session_state.get("override_key"):
        is_valid, err = validate_openrouter_key(st.session_state["override_key"])
        st.session_state["key_valid"] = is_valid
        st.session_state["key_error"] = err
        if is_valid:
            st.success("API key verified with OpenRouter.")
        else:
            st.error(f"API key invalid: {err}")

if st.session_state.get("override_key"):
    OPENROUTER_API_KEY = st.session_state["override_key"]
if st.session_state.get("k"):
    MAX_SNIPPETS = int(st.session_state["k"])
if st.session_state.get("max_chars"):
    MAX_SNIPPET_CHARS = int(st.session_state["max_chars"])

# Load FAQ artifacts from PDF only
faq_pages, faq_matrix, vectorizer = load_faq_artifacts(FAQ_PDF_PATH)

selected_q = st.session_state.get("selected_question", "")
query = st.text_input(
    "Your question",
    value=selected_q,
    placeholder="e.g., What is Trent Limited and what does the company do?",
    key="question",
)

question_answered = st.session_state.get("question_answered", False)

col1, col2 = st.columns([4, 1])
with col1:
    if st.button("Ask") or query:
        if not query:
            st.info("Type a question above.")
        else:
            key = get_api_key()
            if not key:
                st.error("OpenRouter API key missing. Add it in the sidebar or via secrets/environment.")
                st.stop()
            if st.session_state.get("override_key") and not st.session_state.get("key_valid"):
                st.error(f"OpenRouter API key not valid: {st.session_state.get('key_error','Unknown error')}")
                st.stop()

            if faq_matrix is not None and faq_pages:
                context = retrieve_context(query, vectorizer, faq_matrix, faq_pages, k=MAX_SNIPPETS)
                sys_p, user_p = build_prompt(query, context)
                try:
                    answer = call_openrouter(key, MODEL_NAME, sys_p, user_p)
                    st.subheader("Answer")
                    st.write(answer)
                    st.subheader("Context used")
                    for s in context:
                        st.markdown(f"**FAQ Page {s['page_number']}** · score {s['score']:.3f}")
                        st.write(s["text"])
                        st.divider()
                    st.session_state["question_answered"] = True
                except requests.HTTPError as e:
                    st.error(f"OpenRouter API error: {e}\n{e.response.text if e.response is not None else ''}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
            else:
                st.error("FAQ PDF data not loaded properly. Please check the PDF file.")

with col2:
    st.empty()

# --- Quick FAQs ---
if not question_answered:
    st.markdown("### Quick FAQs")
    _faq_cols = st.columns([1, 1, 1, 1, 1, 1])
    _faq_targets = TOP_FAQ_QUESTIONS[:6]
    for idx, q in enumerate(_faq_targets):
        with _faq_cols[idx]:
            if st.button(q, key=f"quick_faq_{idx}", use_container_width=True):
                st.session_state["selected_question"] = q
                st.rerun()
else:
    st.markdown("### Quick FAQs")
    if st.button("🔄 Reset - Show Quick FAQs Again", key="reset_faqs", use_container_width=True):
        st.session_state["question_answered"] = False
        st.session_state["selected_question"] = ""
        st.rerun()

# --- Footer ---
st.markdown(
    """
<hr/>
<div style="text-align:center; opacity:0.7; font-size:14px;">
    <span>© 2025 Trent/Zudio FAQ Chatbot (PDF) · Built with Streamlit</span>
  </div>
""",
    unsafe_allow_html=True,
)


