# DocAnalyzer - NLP Document Analyzer

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An NLP Document Analyzer powered by **Google Gemini** that extracts insights, summaries, entities, topics, and sentiment from documents. Features an interactive Streamlit dashboard with a built-in document Q&A chat interface.

---

## Features

| Feature | Description |
|---------|-------------|
| **Summarization** | Executive summary with key bullet points |
| **Entity Extraction** | Identifies people, organizations, locations, and dates |
| **Sentiment Analysis** | Overall sentiment, tone classification, and explanation |
| **Topic Modeling** | Extracts main topics with descriptions |
| **Interactive Q&A** | Chat with your document and get context-aware answers |
| **Multi-format Support** | Handles both PDF and TXT documents |

---

## Architecture

```
+-----------------------------------------------------+
|                   Streamlit UI (app.py)              |
|  +-----------+  +----------+  +------------------+  |
|  |  Upload & |  | Analysis |  |   Q&A Chat       |  |
|  |  Config   |  | Dashboard|  |   Interface      |  |
|  +-----+-----+  +----+-----+  +--------+---------+  |
+--------+--------------+----------------+-------------+
         |              |                |
    +----v--------------v----------------v---------+
    |           DocumentAnalyzer (analyzer.py)      |
    |   summarize | entities | sentiment | topics   |
    |   ask_question | full_analysis                |
    +-------+----------------+-----------+----------+
            |                |
   +--------v-----+  +------v----------+
   | GeminiClient |  |  PromptManager  |
   |(llm_client)  |  |(prompt_manager) |
   |              |  |                 |
   | - API calls  |  | - Load .txt     |
   | - Retry      |  | - Render vars   |
   | - Rate limit |  | - Cache         |
   +--------------+  +-----------------+
            |
   +--------v----------+
   | DocumentProcessor  |
   |(document_processor)|
   |                    |
   | - PDF extraction   |
   | - TXT reading      |
   | - Text chunking    |
   | - Metadata         |
   +--------------------+
```

---

## Prerequisites

- **Python 3.9** or higher
- **Gemini API Key** (free, no credit card required)
  Get yours at [Google AI Studio](https://aistudio.google.com/apikey)

---

## Setup

### 1. Clone or download the project

```bash
cd "NLP project"
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
# Edit .env and paste your Gemini API key
```

Alternatively, you can enter the API key directly in the app sidebar.

---

## Usage

```bash
streamlit run app.py
```

Then:
1. Enter your **Gemini API Key** in the sidebar (or set it in `.env`)
2. **Upload** a PDF or TXT document
3. Click **Analyze Document**
4. Explore results across five tabs: Summary, Entities, Sentiment, Topics, Q&A
5. Use the **Q&A tab** to ask follow-up questions about your document

---

## Project Structure

```
NLP project/
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── README.md                      # This file
│
├── prompts/                       # Externalized LLM prompt templates
│   ├── summarize.txt              # Document summarization
│   ├── extract_entities.txt       # Named entity extraction
│   ├── sentiment.txt              # Sentiment and tone analysis
│   ├── topics.txt                 # Topic modeling
│   └── qa.txt                     # Question-answering (with CoT)
│
├── src/                           # Core source modules
│   ├── __init__.py
│   ├── llm_client.py              # Gemini API wrapper with retry logic
│   ├── document_processor.py      # PDF/TXT parsing, chunking, metadata
│   ├── prompt_manager.py          # Prompt template loading and rendering
│   └── analyzer.py                # NLP analysis orchestrator
│
└── tests/                         # Unit tests
    ├── __init__.py
    ├── test_document_processor.py  # Tests for document processing
    └── test_prompt_manager.py      # Tests for prompt management
```

---

## Prompt Engineering Approach

Prompts are **externalized** into `.txt` files in the `prompts/` directory, completely decoupled from application logic. This design enables:

- **Easy tuning** - modify prompt instructions without touching Python code
- **Structured output** - all prompts request JSON responses with defined schemas
- **Guardrails** - each prompt includes anti-hallucination constraints
- **Few-shot examples** - embedded examples guide the LLM output format
- **Template variables** - dynamic content (`{document_text}`, `{user_question}`) is injected at runtime via `str.format_map()`
- **XML delimiters** - `<instructions>`, `<document_text>`, `<output_format>` tags clearly separate prompt sections

### Prompt Template Example

```
You are an expert document analyst...

<instructions>
1. Read the document carefully.
2. Generate a concise summary.
...
</instructions>

<document_text>
{document_text}
</document_text>

<output_format>
Return a JSON object: {{"summary": "...", "key_points": [...]}}
</output_format>
```

---

## Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | [Google Gemini](https://ai.google.dev/) | Text analysis via API |
| **API SDK** | [`google-genai`](https://pypi.org/project/google-genai/) | Gemini API integration |
| **Web UI** | [Streamlit](https://streamlit.io/) | Interactive dashboard |
| **PDF Processing** | [pdfplumber](https://github.com/jsvine/pdfplumber) | PDF text extraction |
| **Testing** | [pytest](https://docs.pytest.org/) | Automated testing |

---

## License

This project is licensed under the MIT License.
