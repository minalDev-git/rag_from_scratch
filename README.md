# RAG From Scratch - LangChain Series Learnings

This repository contains implementations of various RAG (Retrieval-Augmented Generation) techniques as taught in the LangChain "RAG From Scratch" series on YouTube. Each file represents one lecture and delivers one specific concept.

## Lectures

1. **[Basic RAG Pipeline](rag_pipeline.py)**  
   Building a foundational RAG system: loading documents, splitting text, creating embeddings, storing in a vector database, retrieving relevant context, and generating answers with an LLM.

2. **[Query Translation: Multi-query](qt_multi_query.py)**  
   Generating multiple perspectives of a user question to overcome limitations of distance-based similarity search. The technique creates alternative questions and retrieves documents for each, then combines results.

3. **[Query Translation: HyDE](qt_hyde.py)**  
   Using Hypothetical Document Embeddings (HyDE): generating a fictional document (passage) that answers the question, then using that passage to retrieve relevant real documents.

4. **[Query Translation: RAG Fusion](qt_rag_fusion.py)**  
   Implementing RAG-Fusion: generating multiple search queries from a single input, retrieving documents for each query, and combining results using reciprocal rank fusion (RRF) to rerank documents.

5. **[Query Translation: Decomposition](qt_decomposition.py)**  
   Breaking down complex questions into simpler sub-questions (query translation via decomposition). The file includes a framework for IR-COT (Iterative Retrieval-Chain of Thought) and Least-most RAG (commented out for extension).

6. **[Query Translation: Step-back Prompting](qt_step_back_prompt.py)**  
   Using step-back prompting to first generate a more generic, easier-to-answer question, retrieve context for both the original and step-back questions, then synthesize a final answer using both contexts.

## How to Use

Each script is self-contained and can be run independently:

```bash
python rag_pipeline.py
python qt_multi_query.py
# ... and so on
```

Ensure you have the required dependencies installed (see `requirements.txt` or the imports in each file) and set up your environment variables (e.g., `GROQ_API_KEY`).

## Learning Path

Follow the files in order to progressively build your understanding of advanced RAG techniques, starting from a basic pipeline and moving through various query translation strategies that improve retrieval quality and answer generation.

---

_Created while following the LangChain "RAG From Scratch" series._
