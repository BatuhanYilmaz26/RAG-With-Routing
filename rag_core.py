"""Core backend logic for the RAG app: models, ingestion, routing, retrieval, web fallback.

This module intentionally contains no Streamlit page setup or UI layout; it interacts with
Streamlit session state and shows minimal status messages when appropriate.
"""

from __future__ import annotations

import os
import tempfile
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# LLM + Embeddings (Gemini preferred; fallback to sentence-transformers embeddings if no API key)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings

# Fallback web search via LangGraph ReAct agent
from langgraph.prebuilt import create_react_agent
# Prefer the renamed package `ddgs`; fall back to the old import if present
try:
    from ddgs import DDGS  # type: ignore
except Exception:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception:
        DDGS = None  # type: ignore

# Qdrant (cloud)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from pypdf import PdfReader
import pytesseract
from PIL import Image

# Optional PDF rendering and OCR backends (no external paths required)
import importlib

try:
    fitz = importlib.import_module("fitz")  # PyMuPDF
except Exception:
    fitz = None  # type: ignore
try:
    easyocr = importlib.import_module("easyocr")  # Pure-Python OCR fallback (uses PyTorch)
except Exception:
    easyocr = None  # type: ignore


DatabaseType = Literal["products", "support", "finance"]


@dataclass
class CollectionConfig:
    name: str
    description: str
    collection_name: str  # Qdrant collection name


COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "products": CollectionConfig(
        name="Product Information",
        description="Product details, specifications, and features",
        collection_name="products_collection",
    ),
    "support": CollectionConfig(
        name="Customer Support & FAQ",
        description="Customer support information, frequently asked questions, and guides",
        collection_name="support_collection",
    ),
    "finance": CollectionConfig(
        name="Financial Information",
        description="Financial data, revenue, costs, and liabilities",
        collection_name="finance_collection",
    ),
}


def _embedding_dimension(emb) -> int:
    try:
        return len(emb.embed_query("dimension-probe"))
    except Exception:
        return 768  # reasonable default for Gemini embeddings


def initialize_models() -> bool:
    """Initialize LLM, embeddings, Qdrant client and collections.

    Returns True when ready.
    """
    if not (st.session_state.qdrant_url and st.session_state.qdrant_api_key):
        return False

    # LLM and Embeddings
    emb = None
    llm = None

    if st.session_state.google_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
        # Gemini embeddings + chat
        try:
            emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        except Exception as e:
            st.warning(f"Falling back to local embeddings due to Gemini error: {e}")

    if emb is None:
        # Local embeddings fallback (no LLM fallback available)
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.info("Using local sentence-transformers embeddings. Provide a Google API key to enable the LLM.")

    # Qdrant client
    try:
        client = QdrantClient(url=st.session_state.qdrant_url, api_key=st.session_state.qdrant_api_key)
        client.get_collections()  # Test connection
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        return False

    # Ensure collections exist with correct vector size; if mismatch, use a suffixed collection name
    def _qdrant_collection_dim(c: QdrantClient, name: str) -> Optional[int]:
        try:
            info = c.get_collection(name)
            vectors = info.config.params.vectors  # can be VectorParams or dict of named vectors
            # Single vector
            if hasattr(vectors, "size"):
                return vectors.size  # type: ignore[attr-defined]
            # Named vectors
            if isinstance(vectors, dict) and vectors:
                first = next(iter(vectors.values()))
                return getattr(first, "size", None)
        except Exception:
            return None
        return None

    vector_size = _embedding_dimension(emb)
    try:
        existing = {c.name for c in client.get_collections().collections}
        databases: Dict[DatabaseType, QdrantVectorStore] = {}
        for db_type, cfg in COLLECTIONS.items():
            target_name = cfg.collection_name
            if target_name in existing:
                existing_dim = _qdrant_collection_dim(client, target_name)
                if existing_dim is not None and existing_dim != vector_size:
                    # Create/use a dimension-suffixed collection to avoid errors
                    target_name = f"{cfg.collection_name}_{vector_size}"
            # Create target collection if it doesn't exist
            if target_name not in existing:
                client.create_collection(
                    collection_name=target_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                existing.add(target_name)

            # Build vector store for the chosen collection
            databases[db_type] = QdrantVectorStore(
                client=client,
                collection_name=target_name,
                embedding=emb,
            )
    except Exception as e:
        st.error(f"Failed to prepare Qdrant collections: {e}")
        return False

    # Save in session
    st.session_state.embeddings = emb
    st.session_state.llm = llm
    st.session_state.qdrant_client = client
    st.session_state.databases = databases
    return True


def process_pdfs(uploaded_files: List[Any]) -> List[Document]:
    """Load and split PDF files into LangChain Documents."""
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            # Read metadata/title and first-page header
            pdf_title = None
            first_header = None
            try:
                reader = PdfReader(tmp_path)
                pdf_title = getattr(getattr(reader, "metadata", None), "title", None)
                first_page_text = ""
                if reader.pages:
                    first_page_text = reader.pages[0].extract_text() or ""
                    lines = [ln.strip() for ln in first_page_text.splitlines() if ln and ln.strip()]
                    first_header = " \u2022 ".join(lines[:3]) if lines else None
            except Exception:
                pass

            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()
            # Optional OCR enhancement for pages with little/no text
            try:
                if st.session_state.ocr_enabled:
                    needs_ocr = any(len((d.page_content or "").strip()) < 50 for d in file_docs)
                    if needs_ocr:
                        # Prefer PyMuPDF to render pages (no Poppler dependency)
                        images: List[Image.Image] = []
                        try:
                            if fitz is None:
                                raise RuntimeError("PyMuPDF not available")
                            pdfdoc = fitz.open(tmp_path)
                            for p in range(pdfdoc.page_count):
                                page = pdfdoc.load_page(p)
                                pix = page.get_pixmap(dpi=200)
                                mode = "RGBA" if pix.alpha else "RGB"
                                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                                images.append(img.convert("RGB"))
                            pdfdoc.close()
                        except Exception:
                            # As a last resort, attempt pdf2image without requiring a user-specified Poppler path
                            try:
                                from pdf2image import convert_from_path  # lazy import

                                images = convert_from_path(tmp_path, dpi=200)
                            except Exception:
                                images = []

                        # Determine OCR engine automatically: prefer Tesseract if available else EasyOCR
                        use_tesseract = False
                        try:
                            _ = pytesseract.get_tesseract_version()
                            use_tesseract = True
                            st.session_state.ocr_engine = "tesseract"
                        except Exception:
                            st.session_state.ocr_engine = None

                        # Initialize EasyOCR reader if needed
                        if not use_tesseract and easyocr is not None:
                            if st.session_state.ocr_reader is None:
                                try:
                                    st.session_state.ocr_reader = easyocr.Reader(["en"], gpu=False)
                                except Exception:
                                    st.session_state.ocr_reader = None
                            if st.session_state.ocr_reader is not None:
                                st.session_state.ocr_engine = "easyocr"

                        for i, img in enumerate(images):
                            # Optional OpenCV pre-processing
                            pre_img = img
                            if st.session_state.get("use_opencv_preproc", False):
                                try:
                                    import cv2
                                    import numpy as np

                                    cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
                                    gray = cv2.bilateralFilter(gray, 9, 75, 75)
                                    th = cv2.adaptiveThreshold(
                                        gray,
                                        255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        31,
                                        15,
                                    )
                                    pre_img = Image.fromarray(th)
                                except Exception:
                                    pass

                            ocr_text = ""
                            if st.session_state.ocr_engine == "tesseract":
                                try:
                                    ocr_text = pytesseract.image_to_string(pre_img, config="--oem 3 --psm 6")
                                except Exception:
                                    ocr_text = ""
                            elif st.session_state.ocr_engine == "easyocr" and st.session_state.ocr_reader is not None:
                                try:
                                    import numpy as np

                                    res = st.session_state.ocr_reader.readtext(np.array(pre_img))
                                    ocr_text = "\n".join([t[1] for t in res]) if res else ""
                                except Exception:
                                    ocr_text = ""

                            if i < len(file_docs) and ocr_text.strip():
                                orig = file_docs[i].page_content or ""
                                if len(ocr_text.strip()) > len(orig.strip()):
                                    file_docs[i] = Document(
                                        page_content=ocr_text,
                                        metadata=file_docs[i].metadata,
                                    )
            except Exception as e:
                st.info(f"OCR skipped due to: {e}")
            os.unlink(tmp_path)
            # Augment per-page metadata and also add a synthetic metadata doc per file
            augmented: List[Document] = []
            for d in file_docs:
                meta = d.metadata or {}
                meta["file_name"] = getattr(file, "name", None) or meta.get("source")
                if pdf_title:
                    meta["pdf_title"] = pdf_title
                # Capture per-page header as first non-empty few lines of that page
                try:
                    plines = [ln.strip() for ln in (d.page_content or "").splitlines() if ln and ln.strip()]
                    if plines:
                        meta["page_header"] = " • ".join(plines[:3])
                except Exception:
                    pass
                augmented.append(Document(page_content=d.page_content, metadata=meta))

            # Synthetic file-level metadata doc to improve retrieval for "title/file name" questions
            meta_lines = [
                "FILE_METADATA",
                f"File Name: {getattr(file, 'name', '')}",
                f"PDF Title: {pdf_title or ''}",
                f"First Page Header: {first_header or ''}",
            ]
            augmented.append(
                Document(
                    page_content="\n".join(meta_lines),
                    metadata={
                        "file_name": getattr(file, "name", None),
                        "doc_type": "file_metadata",
                    },
                )
            )

            chunks = splitter.split_documents(augmented)
            docs.extend(chunks)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
    return docs


def ingest_documents(db_type: DatabaseType, docs: List[Document]) -> int:
    """Add documents to the corresponding Qdrant collection."""
    if not docs:
        return 0
    # Tag docs with source db
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.update({"db_type": db_type})

    vs = st.session_state.databases.get(db_type)
    if vs is None:
        st.error("Vector store not initialized.")
        return 0
    try:
        vs.add_documents(docs)
        return len(docs)
    except Exception as e:
        st.error(f"Failed to ingest documents: {e}")
        return 0


def _is_list_aggregation_query(question: str) -> bool:
    """Heuristically detect queries asking to list/aggregate items (projects, skills, etc.)."""
    q = (question or "").lower()
    keywords = [
        "project",
        "projects",
        "skill",
        "skills",
        "certification",
        "certifications",
        "award",
        "awards",
        "publication",
        "publications",
        "responsibilities",
        "technolog",
        "tools",
        "courses",
    ]
    return any(k in q for k in keywords)


def _looks_like_titles_only(answer: str) -> bool:
    """Detect if the answer is mostly a list of bare titles without descriptions."""
    if not answer:
        return False
    lines = [ln.strip() for ln in str(answer).splitlines() if ln and ln.strip()]
    if not lines:
        return False
    bullet_like = []
    for ln in lines:
        starts_bullet = ln.startswith(("-", "*", "•")) or bool(re.match(r"^\d+[\).\s]", ln))
        if starts_bullet:
            bullet_like.append(ln)
    if len(bullet_like) < 2:
        return False
    short_bare = 0
    for ln in bullet_like:
        # Remove leading bullets/enumeration
        clean = re.sub(r"^([-*•\s]|\d+[\).\s])+", "", ln).strip()
        # Short if few words and lacks punctuation/connector
        words = clean.split()
        if len(words) <= 7 and not any(p in clean for p in [",", ".", ":", " - ", " — "]):
            short_bare += 1
    return short_bare / max(1, len(bullet_like)) >= 0.6


def _bullet_count(answer: str) -> int:
    """Count bullet-like lines in an answer."""
    if not answer:
        return 0
    lines = [ln.strip() for ln in str(answer).splitlines() if ln and ln.strip()]
    cnt = 0
    for ln in lines:
        if ln.startswith(("-", "*", "•")) or bool(re.match(r"^\d+[\).\s]", ln)):
            cnt += 1
    return cnt


def _vector_route(question: str) -> Tuple[Optional[DatabaseType], Dict[str, float]]:
    """Pick DB with best normalized relevance score across collections. Return (db, scores)."""
    scores: Dict[str, float] = {}
    best_db: Optional[DatabaseType] = None
    best_score: float = float("-inf")

    # Exclude synthetic FILE_METADATA chunks from routing to avoid generic overlaps
    routing_filter: Optional[Filter] = Filter(
        must_not=[FieldCondition(key="doc_type", match=MatchValue(value="file_metadata"))]
    )

    for db_type, vs in st.session_state.databases.items():
        try:
            # Returns normalized relevance scores (higher is better, typically 0..1)
            results = vs.similarity_search_with_relevance_scores(
                question, k=5, filter=routing_filter
            )
            if results:
                # Use average of top-3 scores for robustness
                top_scores = sorted([float(s) for (_, s) in results], reverse=True)[:3]
                avg_score = sum(top_scores) / max(1, len(top_scores))
                scores[db_type] = avg_score
                if avg_score > best_score:
                    best_score = avg_score
                    best_db = db_type  # type: ignore
        except Exception:
            # Skip collection on error
            continue

    # Calibrated threshold/margin: favor in-domain routing; still avoid OOD
    THRESHOLD = 0.7
    if st.session_state.get("llm") is None:
        # When no LLM fallback exists, allow slightly lower threshold to keep RAG useful
        THRESHOLD = 0.65
    MARGIN = 0.05  # require separation from runner-up

    if best_db is None:
        return None, scores

    # Check margin only if we have at least two candidates
    margin_ok = True
    if len(scores) >= 2:
        sorted_scores = sorted(scores.values(), reverse=True)
        top = sorted_scores[0]
        second = sorted_scores[1]
        margin_ok = (top - second) >= MARGIN

    if best_score >= THRESHOLD and margin_ok:
        return best_db, scores
    return None, scores


def _llm_route(question: str) -> Optional[DatabaseType]:
    """Use LLM to decide 'products' | 'support' | 'finance' | 'none'"""
    if st.session_state.llm is None:
        return None
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    """
You are a routing assistant. Classify the user's question into exactly one of: products, support, finance.
If none apply, return: none.
Return only the label, with no explanation or punctuation.
"""
                ).strip(),
            ),
            ("human", "Question: {question}"),
        ]
    )
    chain = prompt | st.session_state.llm
    try:
        result = chain.invoke({"question": question}).content.strip().lower()
        result = result.replace("`", "").replace("'", "").replace('"', "")
        if result in ("products", "support", "finance"):
            return result  # type: ignore
        return None
    except Exception:
        return None


def create_fallback_agent():
    """Create a simple ReAct agent with a DuckDuckGo search tool."""

    def web_search(query: str) -> str:
        """Search the web with DuckDuckGo and return up to 5 concise results.

        Args:
            query: The user query to search for.

        Returns:
            A newline-joined string where each entry contains a title, link, and snippet.
            Returns a short error message if the search fails, or "No results found." if empty.
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                lines = []
                for r in results or []:
                    title = r.get("title") or ""
                    href = r.get("href") or r.get("link") or ""
                    snippet = r.get("body") or r.get("snippet") or ""
                    lines.append(f"- {title}\n  {href}\n  {snippet}")
                return "\n".join(lines) if lines else "No results found."
        except Exception as e:
            return f"Search failed: {e}"

    tools = [web_search]

    # Prefer Gemini if available; else this will not run.
    if st.session_state.llm is None:
        return None
    agent = create_react_agent(model=st.session_state.llm, tools=tools, debug=False)
    return agent


def query_database(db: QdrantVectorStore, question: str) -> Tuple[str, List[Document]]:
    """RAG over a specific DB."""
    # Use broader retrieval by default; if list-style query (e.g., projects), broaden further
    is_list_q = _is_list_aggregation_query(question)
    base_kwargs = {"k": 20, "fetch_k": 80, "lambda_mult": 0.25}
    list_kwargs = {"k": 28, "fetch_k": 120, "lambda_mult": 0.3}
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs=list_kwargs if is_list_q else base_kwargs,
    )
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        raise ValueError("No relevant documents found.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    """
You are a helpful AI assistant. Answer the user's question using only the provided context.
If the context is insufficient, say so briefly. Be concise and factual.
When available, you may also rely on metadata present in the context, such as file_name, pdf_title, page_header,
or any FILE_METADATA chunks that summarize a document. If the user asks for a document title or file name, look for
these fields explicitly before concluding it's missing.

If multiple distinct relevant items exist in the context (e.g., projects, skills, certifications, tools, responsibilities),
enumerate ALL of them across pages. For EACH item, include a brief 1–2 sentence description using details from the context
(e.g., purpose, key responsibilities, technologies, outcomes/impact). Prefer a concise bullet list in the form:
- Name — description (tech stack, impact)
Use only information present in the context; do not invent facts.

If the provided context is insufficient to answer the user's question, respond with exactly the single word:
INSUFFICIENT
"""
                ).strip(),
            ),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
        ]
    )

    if st.session_state.llm is None:
        # No LLM available
        joined = "\n---\n".join(d.page_content[:300] for d in relevant_docs[:3])
        return (
            "LLM not configured. Top context snippets shown instead:\n\n" + joined,
            relevant_docs,
        )

    combine_docs = create_stuff_documents_chain(st.session_state.llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs)
    result = rag_chain.invoke({"input": question})
    answer = result.get("answer") or result.get("result") or "No answer produced."

    # Adaptive enhancement: if the answer looks like titles-only OR not enough items for a list query,
    # force a description-enriched pass with broader retrieval
    need_enrich = _looks_like_titles_only(answer)
    if not need_enrich and is_list_q and _bullet_count(answer) < 3:
        need_enrich = True

    if need_enrich and st.session_state.llm is not None:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 32, "fetch_k": 128, "lambda_mult": 0.25},
        )
        relevant_docs = retriever.invoke(question)

        enrich_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        """
You are a helpful AI assistant. Produce a comprehensive bullet list where EACH item includes a 1–2 sentence description
drawn strictly from the context (purpose, responsibilities, technologies, outcomes/impact). Use the form:
- Name — description (tech stack, impact)
Enumerate ALL distinct items mentioned; do not merge multiple projects into a single bullet.
Do not invent facts. If details are missing, say "(details not present)" for that part.

If the provided context is insufficient to answer the user's question, respond with exactly the single word:
INSUFFICIENT
"""
                    ).strip(),
                ),
                ("human", "Context:\n{context}\n\nQuestion: {input}"),
            ]
        )

        combine_docs = create_stuff_documents_chain(st.session_state.llm, enrich_prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs)
        result = rag_chain.invoke({"input": question})
        answer = result.get("answer") or result.get("result") or answer
    # If the model declares the context insufficient, escalate to caller via exception
    if isinstance(answer, str):
        a = answer.strip()
        if a.upper() == "INSUFFICIENT" or any(
            p in a.lower()
            for p in [
                "context is insufficient",
                "insufficient",
                "not enough context",
                "not present in the provided context",
                "no information in the provided context",
            ]
        ):
            raise ValueError("INSUFFICIENT_LOCAL_CONTEXT")
    return answer, relevant_docs


def query_across_databases(question: str) -> Tuple[str, List[Document]]:
    """Aggregate retrieval across all databases and answer from combined context.

    Returns (answer, sources). Raises ValueError if no relevant documents found.
    """
    all_hits: List[Tuple[Document, float, str]] = []  # (doc, score, db_type)
    is_list_q = _is_list_aggregation_query(question)
    for db_type, vs in st.session_state.databases.items():
        try:
            k = 10 if is_list_q else 6
            results = vs.similarity_search_with_relevance_scores(question, k=k)
            for doc, score in results or []:
                all_hits.append((doc, float(score), db_type))
        except Exception:
            continue

    if not all_hits:
        raise ValueError("No relevant documents found across databases.")

    # Keep top-N by score, apply a modest relevance floor
    all_hits.sort(key=lambda x: x[1], reverse=True)
    filtered = [(d, s, t) for (d, s, t) in all_hits if s >= 0.4]
    top_cap = 14
    backfill_cap = 10
    top = filtered[:top_cap] if filtered else all_hits[:backfill_cap]

    # Deduplicate by source + page header snippet to avoid repeats
    seen = set()
    docs: List[Document] = []
    for d, _, _ in top:
        meta = d.metadata or {}
        key = (meta.get("source"), meta.get("file_name"), meta.get("page_header"))
        if key in seen:
            continue
        seen.add(key)
        docs.append(d)

    if st.session_state.llm is None:
        joined = "\n---\n".join(d.page_content[:300] for d in docs[:3])
        return (
            "LLM not configured. Top cross-database context snippets shown instead:\n\n" + joined,
            docs,
        )

    # Build a compact context from the top docs
    def _doc_summary(d: Document) -> str:
        meta = d.metadata or {}
        parts = []
        if meta.get("file_name"):
            parts.append(f"File: {meta.get('file_name')}")
        if meta.get("pdf_title"):
            parts.append(f"Title: {meta.get('pdf_title')}")
        if meta.get("page_header"):
            parts.append(f"Header: {meta.get('page_header')}")
        header = " | ".join(parts)
        body = (d.page_content or "").strip()
        return (header + "\n" + body) if header else body

    max_docs = 16 if is_list_q else 12
    context = "\n\n".join(_doc_summary(d)[:1800] for d in docs[:max_docs])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    """
You are a helpful AI assistant. Answer the user's question using only the provided context.
If the context is insufficient, say so briefly. Be concise and factual.
Use any metadata (file_name, pdf_title, page_header, FILE_METADATA) to ground your answer.
If multiple distinct relevant items exist in the context (e.g., projects, skills, certifications, tools, responsibilities),
enumerate ALL of them across pages. For EACH item, include a brief 1–2 sentence description using details from the context
(purpose, responsibilities, technologies, outcomes/impact). Prefer a concise bullet list: "Name — description".
Do not invent facts.

If the provided context is insufficient to answer the user's question, respond with exactly the single word:
INSUFFICIENT
"""
                ).strip(),
            ),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
        ]
    )
    chain = prompt | st.session_state.llm
    out = chain.invoke({"context": context, "input": question})
    answer = getattr(out, "content", str(out)) or "No answer produced."

    # Adaptive enhancement: if the answer looks like titles-only OR not enough items for a list query,
    # re-run with an enrichment prompt using the built context
    need_enrich = _looks_like_titles_only(answer)
    if not need_enrich and is_list_q and _bullet_count(answer) < 3:
        need_enrich = True

    if need_enrich and st.session_state.llm is not None:
        # Build a more detailed prompt focusing on descriptions
        enrich_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        """
You are a helpful AI assistant. Produce a comprehensive bullet list where EACH item includes a 1–2 sentence description
drawn strictly from the context (purpose, responsibilities, technologies, outcomes/impact). Use the form:
- Name — description (tech stack, impact)
Enumerate ALL distinct items mentioned; do not merge multiple projects into a single bullet.
Do not invent facts. If details are missing, say "(details not present)" for that part.

If the provided context is insufficient to answer the user's question, respond with exactly the single word:
INSUFFICIENT
"""
                    ).strip(),
                ),
                ("human", "Context:\n{context}\n\nQuestion: {input}"),
            ]
        )
        chain2 = enrich_prompt | st.session_state.llm
        out2 = chain2.invoke({"context": context, "input": question})
        answer2 = getattr(out2, "content", str(out2))
        if answer2:
            answer = answer2
    # If the model declares the context insufficient, escalate to caller via exception
    if isinstance(answer, str):
        a = answer.strip()
        if a.upper() == "INSUFFICIENT" or any(
            p in a.lower()
            for p in [
                "context is insufficient",
                "insufficient",
                "not enough context",
                "not present in the provided context",
                "no information in the provided context",
            ]
        ):
            raise ValueError("INSUFFICIENT_LOCAL_CONTEXT")
    return answer, docs


def handle_web_fallback(question: str) -> Tuple[str, List[Document]]:
    st.info("No relevant documents found or routing uncertain. Using web research fallback…")

    def _direct_duckduckgo(q: str, max_results: int = 8) -> str:
        if DDGS is None:
            return "Search failed: ddgs client not installed. Install 'ddgs' package."
        try:
            with DDGS() as ddgs_client:  # type: ignore
                results = ddgs_client.text(q, max_results=max_results)
                lines: List[str] = []
                for r in results or []:
                    title = r.get("title") or ""
                    href = r.get("href") or r.get("link") or ""
                    snippet = r.get("body") or r.get("snippet") or ""
                    if title or href or snippet:
                        lines.append(f"- {title}\n  {href}\n  {snippet}")
                return "\n".join(lines) if lines else "No results found."
        except Exception as e:
            return f"Search failed: {e}"

    def _extract_links(results: str, max_links: int = 3) -> List[str]:
        links: List[str] = []
        for line in (results or "").splitlines():
            line = line.strip()
            if line.startswith("http://") or line.startswith("https://"):
                links.append(line)
            elif line.startswith("-") and ("http://" in line or "https://" in line):
                # handle bullet line where link appears after dash
                parts = line.split()
                for p in parts:
                    if p.startswith("http://") or p.startswith("https://"):
                        links.append(p)
                        break
            if len(links) >= max_links:
                break
        return links[:max_links]

    def _summarize_results(q: str, results: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        """
You are a helpful assistant. You will be given web search results (title, link, snippet).
Answer the user's question concisely in 1–4 sentences using only those snippets. Prefer concrete facts present in the snippets.
Avoid refusing. If exact details are unclear, state what's known and any uncertainty, and provide the best available summary grounded in the snippets.
Then add a short Sources section with 1–3 links.
"""
                    ).strip(),
                ),
                (
                    "human",
                    "Question: {q}\n\nWeb results (list):\n{results}\n\nReturn format:\nAnswer: <concise answer>\nSources:\n- <link1>\n- <link2>\n- <link3>\n",
                ),
            ]
        )
        chain = prompt | st.session_state.llm
        ans = chain.invoke({"q": q, "results": results})
        return getattr(ans, "content", str(ans))

    # If no LLM, return raw structured results with links
    if st.session_state.llm is None:
        raw = _direct_duckduckgo(question, max_results=12)
        return (raw if raw else "No results found.", [])

    # With LLM: try summarizing once; if weak, retry with more results
    for max_res in (12, 18):
        raw = _direct_duckduckgo(question, max_results=max_res)
        if not raw or raw.startswith("Search failed") or raw == "No results found.":
            continue
        try:
            summary = _summarize_results(question, raw)
            content = summary or ""
            # Ensure we include sources; append links if missing
            if "Sources:" not in content:
                links = _extract_links(raw, max_links=3)
                if links:
                    src_block = "\nSources:\n" + "\n".join(f"- {u}" for u in links)
                    content = (content + src_block) if content else ("Answer: See sources.\n" + src_block)
            if content.strip():
                return (content, [])
        except Exception:
            continue

    # Final fallback: return raw results with top links
    raw = _direct_duckduckgo(question, max_results=12)
    links = _extract_links(raw, max_links=3)
    if links:
        msg = "Answer: Unable to summarize confidently from snippets. See sources for details.\nSources:\n" + "\n".join(
            f"- {u}" for u in links
        )
        return (msg, [])
    return (raw if raw else "No results found.", [])
