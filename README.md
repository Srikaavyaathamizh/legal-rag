# Legal RAG: Indian Legal Question-Answering System

A Retrieval-Augmented Generation (RAG) based legal question-answering system for Indian criminal law. The system combines temporal-aware legal filtering, BM25 retrieval, semantic search, hybrid ranking, section reranking, and grounded LLM generation to provide evidence-based legal answers.

## Overview

Finding the correct legal provision can be difficult because users may not know the relevant section number or may describe a legal situation using everyday language.

Legal RAG addresses this problem by retrieving relevant legal provisions from a structured legal corpus before generating an answer.

The current implementation focuses on:

* Indian Penal Code (IPC), 1860
* Bharatiya Nyaya Sanhita (BNS), 2023
* Temporal applicability based on incident date

The system is designed so that additional Indian Acts can be added to the legal corpus in future.

## System Architecture

```text
User Question
      |
      v
Query Expansion
      |
      v
Temporal Processing
      |
      +-------------------+
      |                   |
      v                   v
   BM25 Search      Semantic Search
      |                   |
      +---------+---------+
                |
                v
        Hybrid Ranking
                |
                v
        Section Reranking
                |
                v
       Legal Evidence
                |
                v
        Grounded LLM
                |
                v
    Citation / Trust Check
                |
                v
         Final Answer
```

## Key Features

* **Temporal Legal Filtering**
  Uses the incident date to determine whether IPC or BNS provisions are applicable.

* **BM25 Retrieval**
  Performs keyword-based retrieval of relevant legal provisions.

* **Semantic Retrieval**
  Uses sentence embeddings to retrieve provisions based on semantic similarity.

* **Hybrid Retrieval**
  Combines lexical BM25 and semantic retrieval results.

* **Section Reranking**
  Reorders retrieved legal provisions to improve evidence selection.

* **Query Expansion**
  Supports selected legal abbreviations and colloquial terms.

* **Grounded LLM Generation**
  Generates responses using retrieved legal evidence instead of relying only on model knowledge.

* **Citation and Trust Checking**
  Tracks the legal sections used for the generated response and checks grounding.

* **FastAPI Backend**
  Provides REST API endpoints for legal question answering.

## Technology Stack

| Component            | Technology              |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| API Framework        | FastAPI                 |
| Keyword Retrieval    | BM25                    |
| Semantic Retrieval   | Sentence Transformers   |
| Vector Database      | Qdrant                  |
| LLM                  | Groq                    |
| Data Format          | CSV / JSON              |
| Testing              | Pytest                  |
| Frontend             | HTML / CSS / JavaScript |

## Project Structure

```text
legal_rag/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── IPC_dataset.csv
│   │   └── BNS_dataset.csv
│   │
│   └── processed/
│       └── unified_sections.json
│
├── eval/
│
├── frontend/
│
├── notebooks/
│
├── src/
│   ├── temporal_filter.py
│   ├── query_expansion.py
│   ├── bm25_index.py
│   ├── semantic_index.py
│   ├── hybrid_ranker.py
│   ├── section_reranker.py
│   ├── explainability.py
│   ├── llm_explainer.py
│   └── pipeline.py
│
├── tests/
│   ├── test_pipeline.py
│   └── test_api_phase12.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Pipeline Components

### 1. Legal Corpus

The project begins with structured IPC and BNS legal provisions. The processed corpus is stored in:

```text
data/processed/unified_sections.json
```

Each legal record contains information such as the law, section, heading, text, and temporal metadata.

### 2. Temporal Filtering

The system uses the incident date to identify the applicable legal framework.

```text
Incident before 1 July 2024
        |
        v
       IPC

Incident from 1 July 2024
        |
        v
       BNS
```

### 3. BM25 Retrieval

BM25 provides lexical retrieval based on terms appearing in the user's query and the legal provisions.

### 4. Semantic Retrieval

Sentence Transformer embeddings are used to identify provisions that are semantically related to the user's question, even when the exact words differ.

Qdrant is used for persistent vector indexing and similarity search.

### 5. Hybrid Ranking

BM25 and semantic results are combined to improve retrieval quality.

### 6. Section Reranking

The retrieved provisions are further ranked to identify the most relevant legal sections.

### 7. Grounded LLM

The selected legal evidence is passed to the LLM to generate a natural-language response.

### 8. Explainability and Trust

The system records:

* Retrieved legal sections
* Evidence used
* Citation information
* Uncited section flags
* Trustworthiness status

## API

The backend is implemented using FastAPI.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "service": "Legal RAG API"
}
```

### Legal Query

```http
POST /query
```

Example request:

```json
{
  "question": "What is the punishment for wrongful restraint?",
  "incident_date": "2023-06-01"
}
```

The response contains the generated answer along with information about the evidence and trustworthiness of the response.

## Installation

Clone the repository:

```bash
git clone https://github.com/Srikaavyaathamizh/legal-rag.git
cd legal-rag
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Environment Variables

If the project requires an LLM API key, create a `.env` file and configure the required credentials.

Example:

```text
GROQ_API_KEY=your_api_key_here
```

Do not commit API keys or `.env` files to GitHub.

## Running the Backend

From the project root:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Running the Frontend

Open another terminal and run:

```powershell
cd frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

The frontend communicates with the FastAPI backend running on port 8000.

## Testing

Run pipeline tests:

```powershell
python -m pytest tests/test_pipeline.py -q
```

Run API tests:

```powershell
python -m pytest tests/test_api_phase12.py -q
```

The current implementation has successfully passed the existing pipeline and API test suites.

## Current Scope

The current legal corpus focuses on:

* Indian Penal Code (IPC)
* Bharatiya Nyaya Sanhita (BNS)

The system should therefore not be considered a complete database of all Indian laws.

Queries related to Acts that are not included in the corpus may not retrieve the correct legal provision.

## Future Scope

The legal corpus can be expanded to include:

* Bharatiya Nagarik Suraksha Sanhita (BNSS)
* Bharatiya Sakshya Adhiniyam (BSA)
* POCSO Act
* NDPS Act
* Information Technology Act
* Motor Vehicles Act
* Dowry Prohibition Act
* Domestic Violence Act
* Juvenile Justice Act
* Code of Civil Procedure
* Contract Act
* Companies Act
* Consumer Protection Act
* Transfer of Property Act
* Arbitration Act
* Limitation Act

Future versions can also include:

* Multilingual Indian legal Q&A
* Improved legal citation verification
* Case-law and precedent retrieval
* Larger legal knowledge graphs
* Conversation/session history
* More advanced legal reasoning and evidence validation

## Research Contribution

The project focuses on combining multiple retrieval and reasoning components into a unified legal question-answering pipeline:

```text
Temporal Awareness
        +
BM25 Retrieval
        +
Semantic Retrieval
        +
Hybrid Ranking
        +
Section Reranking
        +
Grounded LLM
        +
Citation / Trust Checking
```

This architecture is intended to improve the relevance, traceability, and reliability of generated legal responses.

## Disclaimer

This project is developed for educational and research purposes. Generated responses should not be treated as professional legal advice. Legal information should be verified against authoritative and current legal sources.

## License

See the [LICENSE](LICENSE) file for license information.
