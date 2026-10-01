# RAG From Scratch - LangChain Series Learnings

This repository contains implementations of various RAG (Retrieval-Augmented Generation) techniques as taught in the LangChain "RAG From Scratch" series on YouTube. Each file represents one lecture and delivers one specific concept.

## 📚 Course Curriculum

### Foundation

#### 1. [Basic RAG Pipeline](rag_pipeline.py)

Building a foundational RAG system: loading documents, splitting text, creating embeddings, storing in a vector database, retrieving relevant context, and generating answers with an LLM.

---

### Query Translation Techniques

Query translation improves retrieval by transforming user queries into more effective search formats.

#### 2. [Multi-query](query_translation/qt_multi_query.py)

Generating multiple perspectives of a user question to overcome limitations of distance-based similarity search. The technique creates alternative questions and retrieves documents for each, then combines results.

#### 3. [HyDE (Hypothetical Document Embeddings)](query_translation/qt_hyde.py)

Using Hypothetical Document Embeddings (HyDE): generating a fictional document (passage) that answers the question, then using that passage to retrieve relevant real documents.

#### 4. [RAG Fusion](query_translation/qt_rag_fusion.py)

Implementing RAG-Fusion: generating multiple search queries from a single input, retrieving documents for each query, and combining results using reciprocal rank fusion (RRF) to rerank documents.

#### 5. [Decomposition](query_translation/qt_decomposition.py)

Breaking down complex questions into simpler sub-questions (query translation via decomposition). The file includes a framework for IR-COT (Iterative Retrieval-Chain of Thought) and Least-most RAG (commented out for extension).

#### 6. [Step-back Prompting](query_translation/qt_step_back_prompt.py)

Using step-back prompting to first generate a more generic, easier-to-answer question, retrieve context for both the original and step-back questions, then synthesize a final answer using both contexts.

---

### Routing Techniques

Routing determines which processing path to take based on query characteristics.

#### 7. [Logical Routing](routing/logical_routing.py)

Using logical rules and conditions to route queries for direct classification and conditional execution paths.

#### 8. [Semantic Routing](routing/semantic_routing.py)

Using semantic similarity to classify and route queries with LLM-powered intelligent routing decisions.

---

## 🚀 How to Use

Each script is self-contained and can be run independently:

```bash
# Foundation
python rag_pipeline.py

# Query Translation
python query_translation/qt_multi_query.py
python query_translation/qt_hyde.py
python query_translation/qt_rag_fusion.py
python query_translation/qt_decomposition.py
python query_translation/qt_step_back_prompt.py

# Routing
python routing/logical_routing.py
python routing/semantic_routing.py
```

Ensure you have the required dependencies installed and set up your environment variables (e.g., `GROQ_API_KEY`).

---

_Learning from the LangChain "RAG From Scratch" series on YouTube._
