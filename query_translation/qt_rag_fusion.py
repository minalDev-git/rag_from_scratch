# Query Translation: RAG Fusion

import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="groq/compound",
    temperature=0,
)

# Load blogs
user_agent = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; RAG Bot/1.0; +https://example.com)")
loader = WebBaseLoader(
    web_paths=["https://blog.langchain.com/introducing-langgraph/"],
    requests_kwargs={"headers": {"User-Agent": user_agent}},
)
blog_docs = loader.load()

# Split doc
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
splits = text_splitter.split_documents(blog_docs)
# print(splits)

# Index
vectorstore = Chroma.from_documents(documents=splits,embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever()

# RAG-Fusion: Related
template = """
You are ahelpful assistant that generates multiple search queries based on a simple input query. \n 
Generate multiple search queries related to: {question} \n
Output (4 queries):
"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion 
    | llm 
    | StrOutputParser()
    | (lambda x: x.split('\n'))
)

def reciprocal_rag_fusion(results: list[list], k=60):
    """reciprocal_rag_fusion that takes multiple lists of ranked documents and an optional parameter k used in the RRF formula"""
    # Initialize a dict to hold fused scores for each unique document.
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each each document in the list with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key. (assumes docs can be serialized to JSON)
            doc_str = dumps(doc)
            # If the doc is not yet in the fused_scores dict, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # retrieve the current score of the document, if any
            # prev_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] = 1 / (rank + k)

    # Sort the documents based on there fused scores in decending order to get the final reranked_results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

# Retrieve
question = "What is Langraph?"
retrieval_chain = generate_queries | retriever.map() | reciprocal_rag_fusion

docs = retrieval_chain.invoke({"question": question})
print(len(docs))

# RAG
template = """Answer the question based ONLY on the following context below.

    Context:
    {context}

    Question: {question}

    Provide your answer in at least two lines.
    If the answer is not in the context, respond exactly: "I do not have enough information to answer that."
"""
prompt = ChatPromptTemplate.from_template(template)
final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | prompt 
    | llm 
    | StrOutputParser()
)
response = final_rag_chain.invoke({"question":question})

print(response)