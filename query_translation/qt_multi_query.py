# Query Translation: Multi query

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
    model="groq/compound-mini",
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

# Multi-Query: Different Perspectives
template = """
You are an AI language model assistant. You task is to generate five different versions of the given user question to retrieve relant documents
from a vector database. By generating multiple perspectives on the user question, your goal isto help the user overcome some of the limitations of
distance-based similarity search. Provide these alternative questions separated by newlines. Original question {question}
"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | llm 
    | StrOutputParser()
    | (lambda x: x.split('\n'))
)

def get_unique_union(documents: list[list]):
    """Unique union of relevant docs"""
    # Flatten list of lists, and convert each Document to string
    flatten_docs = [dumps(docs) for sublist in documents for docs in sublist]
    # Get unique documnets
    unique_docs = list(set(flatten_docs))
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is Langraph?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

docs = retrieval_chain.invoke({"question": question})
len(docs)

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