import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="groq/compound-mini",
    temperature=0.7,
)

# Documents
# question = "What kinds of pets do I like?"
# document = "My favourite pet is a cat."

# def num_tokens_from_strings(string: str, encoding_name: str) -> int:
#     """Returns number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# print(num_tokens_from_strings(question,"cl100k_base"))

# query_result = embedding_model.embed_query(question)
# document_result = embedding_model.embed_query(document)

# print(len(query_result))

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

# store splits in index
vectorstore = Chroma.from_documents(documents=splits,embedding=HuggingFaceEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k":1})

question = "What is Langraph?"
docs = retriever.invoke(question)
print(len(docs))

# Create the prompt
template = """Answer the question based ONLY on the following context below.

    Context:
    {context}

    Question: {question}

    Provide your answer in at least two lines.
    If the answer is not in the context, respond exactly: "I do not have enough information to answer that."
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)
response = rag_chain.invoke(question)

print(response)

