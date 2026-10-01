import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
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

# HyDE document generation
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(template)

# Run
generate_docs_for_retrieval = prompt_hyde | llm | StrOutputParser()
question = "What are the main components of Langraph that allow parallel-execution?"
generate_docs_for_retrieval.invoke({"question": question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain.invoke({"question": question})
print(retrieved_docs)

# RAG
template = """Answer the following question based on this context:
Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

res = final_rag_chain.invoke({"context":retrieved_docs, "question": question})
print(res)