"""
DocAnalyzer - Streamlit application for NLP Document Analysis.

Upload a PDF or TXT document, get instant multi-dimensional NLP analysis
(summary, entities, sentiment, topics), and ask follow-up questions
in an interactive chat, powered by Google Gemini.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from src.llm_client import GeminiClient, LLMError
from src.document_processor import DocumentProcessor
from src.prompt_manager import PromptManager
from src.analyzer import DocumentAnalyzer

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocAnalyzer - NLP Document Analyzer",
    page_icon="D",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Ensure all required session-state keys exist."""
    defaults = {
        "analysis_results": None,
        "chat_history": [],
        "document_text": None,
        "metadata": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple:
    """Draw sidebar controls and return (api_key, uploaded_file, analyze_clicked)."""
    st.sidebar.title("DocAnalyzer")
    st.sidebar.markdown("**NLP Document Analyzer** powered by Google Gemini")
    st.sidebar.divider()

    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/apikey",
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Document",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT",
    )

    analyze_clicked = st.sidebar.button(
        "Analyze Document", use_container_width=True
    )

    # Show document metadata in sidebar when available.
    if st.session_state.metadata:
        st.sidebar.divider()
        st.sidebar.markdown("### Document Info")
        meta = st.session_state.metadata
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Words", f"{meta.get('word_count', 0):,}")
        col2.metric("Characters", f"{meta.get('char_count', 0):,}")
        col3, col4 = st.sidebar.columns(2)
        col3.metric("Sentences", meta.get("sentence_count", 0))
        col4.metric("Paragraphs", meta.get("paragraph_count", 0))

    return api_key, uploaded_file, analyze_clicked


# ---------------------------------------------------------------------------
# Analysis display helpers
# ---------------------------------------------------------------------------
def _show_summary(data: dict) -> None:
    """Render the Summary tab."""
    st.header("Executive Summary")
    st.write(data.get("summary", "No summary available."))

    key_points = data.get("key_points", [])
    if key_points:
        st.subheader("Key Points")
        for point in key_points:
            st.markdown(f"- {point}")


def _show_entities(data: dict) -> None:
    """Render the Entities tab."""
    st.header("Named Entities")

    categories = [
        ("persons", "Persons"),
        ("organizations", "Organizations"),
        ("locations", "Locations"),
        ("dates", "Dates"),
    ]
    has_any = False
    cols = st.columns(len(categories))
    for col, (key, label) in zip(cols, categories):
        items = data.get(key, [])
        if items:
            has_any = True
            with col:
                st.markdown(f"**{label}**")
                for item in items:
                    st.markdown(f"- {item}")

    if not has_any:
        st.info("No entities found in the document.")


def _show_sentiment(data: dict) -> None:
    """Render the Sentiment tab."""
    st.header("Sentiment and Tone Analysis")

    sentiment = data.get("sentiment", "Unknown")
    color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
    color = color_map.get(sentiment.lower(), "gray")

    st.markdown(
        f"**Overall Sentiment:** <span style='color:{color}; font-size:1.2em;'>{sentiment}</span>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        tones = data.get("tones", [])
        if tones:
            st.subheader("Tone")
            for tone in tones:
                st.markdown(f"- {tone}")

    with col2:
        explanation = data.get("explanation", "")
        if explanation:
            st.subheader("Explanation")
            st.write(explanation)


def _show_topics(data: dict) -> None:
    """Render the Topics tab."""
    st.header("Topics and Themes")

    topics = data.get("topics", [])
    if topics:
        for t in topics:
            if isinstance(t, dict):
                st.markdown(f"**{t.get('topic_name', 'Topic')}**")
                st.write(t.get("description", ""))
            else:
                st.markdown(f"- {t}")
    else:
        st.info("No topics extracted.")


def _show_qa(api_key: str) -> None:
    """Render the Q&A chat tab."""
    st.header("Document Q&A")
    st.caption("Ask questions about the uploaded document.")

    # Display existing chat history.
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user input.
    if user_input := st.chat_input("Ask a question about the document..."):
        # Show user message immediately.
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if not api_key:
            st.error("Please provide an API key in the sidebar to use Q&A.")
            return

        with st.spinner("Processing..."):
            try:
                client = GeminiClient(api_key=api_key)
                prompt_mgr = PromptManager()
                analyzer = DocumentAnalyzer(client, prompt_mgr)
                result = analyzer.ask_question(
                    text=st.session_state.document_text,
                    question=user_input,
                    chat_history=st.session_state.chat_history[:-1],
                )

                answer = result.get("answer", result.get("raw_response", "Could not generate an answer."))

                # Optionally show chain of thought in an expander.
                reasoning = result.get("chain_of_thought", "")
                display_text = answer
                if reasoning:
                    display_text += f"\n\n<details><summary>Reasoning</summary>\n\n{reasoning}\n\n</details>"

            except LLMError as exc:
                display_text = f"Error: {exc}"
            except Exception as exc:
                display_text = f"Unexpected error: {exc}"

        with st.chat_message("assistant"):
            st.markdown(display_text, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": display_text})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the Streamlit application."""
    _init_session_state()
    api_key, uploaded_file, analyze_clicked = _render_sidebar()

    # ------ Run analysis when button is clicked ------
    if analyze_clicked:
        if not api_key:
            st.sidebar.error("Please provide a Gemini API Key.")
        elif not uploaded_file:
            st.sidebar.error("Please upload a document first.")
        else:
            with st.spinner("Analyzing document, this may take a moment..."):
                try:
                    processor = DocumentProcessor()

                    # Extract text directly from uploaded bytes.
                    file_type = uploaded_file.name.split(".")[-1]
                    document_text = processor.extract_text_from_bytes(
                        uploaded_file.getvalue(), file_type
                    )

                    metadata = processor.get_metadata(document_text)
                    st.session_state.document_text = document_text
                    st.session_state.metadata = metadata

                    # Run full analysis.
                    client = GeminiClient(api_key=api_key)
                    prompt_mgr = PromptManager()
                    analyzer = DocumentAnalyzer(client, prompt_mgr)

                    results = analyzer.full_analysis(document_text)
                    st.session_state.analysis_results = results
                    st.session_state.chat_history = []  # Reset Q&A on new doc.

                    st.sidebar.success("Analysis complete!")
                    st.rerun()  # Rerun so sidebar metrics update.

                except LLMError as exc:
                    st.error(f"LLM Error: {exc}")
                except Exception as exc:
                    st.error(f"Error during analysis: {exc}")

    # ------ Display results (or placeholder) ------
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Summary", "Entities", "Sentiment", "Topics", "Q&A"]
        )

        with tab1:
            _show_summary(results.get("summary", {}))
        with tab2:
            _show_entities(results.get("entities", {}))
        with tab3:
            _show_sentiment(results.get("sentiment", {}))
        with tab4:
            _show_topics(results.get("topics", {}))
        with tab5:
            _show_qa(api_key)
    else:
        # Landing page.
        st.title("DocAnalyzer")
        st.markdown(
            """
            **Welcome to DocAnalyzer** - an NLP document analysis tool.

            ### How to use
            1. Enter your **Gemini API Key** in the sidebar (free at [aistudio.google.com](https://aistudio.google.com/apikey))
            2. **Upload** a PDF or TXT document
            3. Click **Analyze Document**
            4. Explore insights across Summary, Entities, Sentiment, Topics and Q&A tabs

            ### Analysis capabilities
            | Analysis | Description |
            |----------|-------------|
            | Summary | Executive summary with key points |
            | Entities | People, organizations, locations, dates |
            | Sentiment | Overall sentiment, tone and explanation |
            | Topics | Main topics with descriptions |
            | Q&A | Ask follow-up questions about your document |
            """
        )


if __name__ == "__main__":
    main()
