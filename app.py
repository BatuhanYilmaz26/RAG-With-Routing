import asyncio
import warnings
from typing import Dict, Optional, List

import streamlit as st
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from rag_core import (
	COLLECTIONS,
	DatabaseType,
	initialize_models,
	process_pdfs,
	ingest_documents,
	query_across_databases,
	query_database,
	handle_web_fallback,
	_vector_route,
	_llm_route,
)


# ---------- App Config ----------
st.set_page_config(page_title="RAG Agent with Database Routing", page_icon="📠")

# Suppress noisy deprecation warning from a native dependency on shutdown
warnings.filterwarnings(
	"ignore",
	message=r".*swigvarlink.*",
	category=DeprecationWarning,
)


# Ensure an asyncio event loop exists in Streamlit's ScriptRunner thread
def _ensure_event_loop() -> None:
	try:
		asyncio.get_running_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)


_ensure_event_loop()


def init_session_state() -> None:
	if "google_api_key" not in st.session_state:
		st.session_state.google_api_key = ""
	if "qdrant_url" not in st.session_state:
		st.session_state.qdrant_url = ""
	if "qdrant_api_key" not in st.session_state:
		st.session_state.qdrant_api_key = ""
	if "embeddings" not in st.session_state:
		st.session_state.embeddings = None
	if "llm" not in st.session_state:
		st.session_state.llm = None
	if "databases" not in st.session_state:
		# Map[DatabaseType, QdrantVectorStore]
		st.session_state.databases = {}
	if "qdrant_client" not in st.session_state:
		st.session_state.qdrant_client = None
	# OCR settings
	if "ocr_enabled" not in st.session_state:
		st.session_state.ocr_enabled = False
	# OCR engine/runtime state (auto-detected, no user paths required)
	if "ocr_engine" not in st.session_state:
		st.session_state.ocr_engine = None  # "tesseract" | "easyocr" | None
	if "ocr_reader" not in st.session_state:
		st.session_state.ocr_reader = None  # cache EasyOCR reader
	# OpenCV pre-processing toggle
	if "use_opencv_preproc" not in st.session_state:
		st.session_state.use_opencv_preproc = False
	# Web fallback toggle
	if "allow_web_fallback" not in st.session_state:
		st.session_state.allow_web_fallback = True


def render_sidebar():
	st.sidebar.header("Configuration")
	google_api_key = st.sidebar.text_input(
		"Google API Key (Gemini)", type="password", value=st.session_state.google_api_key
	)
	qdrant_url = st.sidebar.text_input(
		"Qdrant URL", value=st.session_state.qdrant_url, help="https://<cluster>.cloud.qdrant.io"
	)
	qdrant_api_key = st.sidebar.text_input(
		"Qdrant API Key", type="password", value=st.session_state.qdrant_api_key
	)

	if google_api_key and google_api_key != st.session_state.google_api_key:
		st.session_state.google_api_key = google_api_key
	if qdrant_url and qdrant_url != st.session_state.qdrant_url:
		st.session_state.qdrant_url = qdrant_url
	if qdrant_api_key and qdrant_api_key != st.session_state.qdrant_api_key:
		st.session_state.qdrant_api_key = qdrant_api_key

	if st.session_state.qdrant_url and st.session_state.qdrant_api_key:
		ready = initialize_models()
		if ready:
			st.sidebar.success("Models and databases initialized.")
		else:
			st.sidebar.warning("Initialization incomplete. Check credentials and try again.")
	else:
		st.sidebar.info("Enter Qdrant credentials to initialize.")

	st.sidebar.markdown("---")

	# Fallback settings
	st.sidebar.subheader("Fallbacks")
	allow_web = st.sidebar.checkbox(
		"Allow web search fallback (DuckDuckGo)",
		value=st.session_state.allow_web_fallback,
		help="When enabled, if no local documents match, the app will use web search to answer.",
	)
	if allow_web != st.session_state.allow_web_fallback:
		st.session_state.allow_web_fallback = allow_web

	# OCR settings
	st.sidebar.subheader("OCR (optional)")
	ocr_enabled = st.sidebar.checkbox("Enable OCR for scanned PDFs", value=st.session_state.ocr_enabled)
	if ocr_enabled != st.session_state.ocr_enabled:
		st.session_state.ocr_enabled = ocr_enabled

	engine = st.session_state.get("ocr_engine")
	if engine == "tesseract":
		st.sidebar.caption("OCR engine: Tesseract (auto-detected)")
	elif engine == "easyocr":
		st.sidebar.caption("OCR engine: EasyOCR (auto)")
	else:
		st.sidebar.caption("OCR engine: none detected yet. Trying Tesseract first, then EasyOCR if available.")

	# OpenCV pre-processing toggle
	if st.session_state.ocr_enabled:
		use_cv = st.sidebar.checkbox(
			"Use OpenCV preprocessing",
			value=st.session_state.use_opencv_preproc,
			help="Improves OCR on noisy/scanned pages with denoise + thresholding",
		)
		if use_cv != st.session_state.use_opencv_preproc:
			st.session_state.use_opencv_preproc = use_cv
	else:
		# Ensure it's off and hide the control when OCR is disabled
		if st.session_state.use_opencv_preproc:
			st.session_state.use_opencv_preproc = False
		st.sidebar.caption("Enable OCR to use OpenCV preprocessing.")


def render_ingestion_ui():
	st.header("Document Upload")
	st.info("Upload PDFs into any database. Each tab maps to a different collection.")
	tabs = st.tabs([cfg.name for cfg in COLLECTIONS.values()])
	for (db_type, cfg), tab in zip(COLLECTIONS.items(), tabs):
		with tab:
			files = st.file_uploader(
				f"Upload PDF files for {cfg.name}",
				type=["pdf"],
				accept_multiple_files=True,
				key=f"uploader_{db_type}",
			)
			if st.button(f"Ingest into {cfg.name}", key=f"ingest_{db_type}"):
				if not files:
					st.warning("Please choose at least one PDF file.")
				else:
					docs = process_pdfs(files)
					count = ingest_documents(db_type, docs)
					st.success(f"Ingested {count} chunks into {cfg.name}.")


def route_query(question: str) -> Optional[DatabaseType]:
	# 1) Vector-first routing
	db, scores = _vector_route(question)
	if db is not None:
		st.success(f"Vector routing selected: {db} (scores: {scores})")
		return db

	st.warning("Low confidence on vector routing. Falling back to LLM routing…")
	# 2) LLM-based routing
	db2 = _llm_route(question)
	if db2 in COLLECTIONS:
		st.success(f"LLM routing selected: {db2}")
		return db2
	st.info("No suitable database from LLM routing. Trying cross-database retrieval, then web search if needed.")
	return None


def render_query_ui():
	st.header("Ask Questions")
	question = st.text_input("Enter your question:")
	if st.button("Ask") and question:
		if not st.session_state.databases:
			st.error("Databases not initialized. Add Qdrant credentials in the sidebar.")
			return
		with st.spinner("Finding answer…"):
			target = route_query(question)
			if target is None:
				# Try cross-database retrieval before web fallback
				try:
					with st.status("Searching all databases…", state="running"):
						answer, sources = query_across_databases(question)
				except Exception:
					# Cross-DB found no relevant docs; optionally use web fallback
					if st.session_state.get("allow_web_fallback", True):
						answer, sources = handle_web_fallback(question)
					else:
						answer, sources = (
							"No relevant local documents found. Web search is disabled in settings.",
							[],
						)
			else:
				try:
					vs = st.session_state.databases[target]
					answer, sources = query_database(vs, question)
				except Exception:
					# If DB query fails (e.g., no docs), try cross-DB before optional web fallback
					try:
						with st.status("Searching all databases…", state="running"):
							answer, sources = query_across_databases(question)
					except Exception:
						if st.session_state.get("allow_web_fallback", True):
							answer, sources = handle_web_fallback(question)
						else:
							answer, sources = (
								"No relevant local documents found. Web search is disabled in settings.",
								[],
							)

		st.subheader("Answer")
		st.write(answer)

		if sources:
			st.subheader("Sources")
			for i, d in enumerate(sources, start=1):
				meta = d.metadata or {}
				src = meta.get("source") or meta.get("file_path") or meta.get("db_type") or "chunk"
				st.markdown(f"{i}. {src}")


def main():
	init_session_state()
	st.title("📠 RAG Agent with Database Routing")
	render_sidebar()
	render_ingestion_ui()
	render_query_ui()


if __name__ == "__main__":
	main()

