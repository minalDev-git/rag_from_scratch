import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

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

# Few shot Examples
# from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "What can the members of The Police do?"
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "What is Jan Sindel's personal history?"
    },
]

# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an expert in world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, 
        which is easier to answer. Here are a few examples: """),

        # Few-shot examples
        few_shot_prompt,

        # New question
        ("user", "{question}")
    ]
)

generic_queries_step_back = prompt | llm | StrOutputParser()
question = "What is task decomposition for LLM agents?"

res = generic_queries_step_back.invoke({"question":question})
print(res)

# Response Prompt
response_prompt_template = """You are an expert of world knowledge.I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
# {normal_context}
# {step_back_context}

Original Question: {question}
Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever, # type: ignore
        # Retrieve context using the step-back question
        "step_back_context": generic_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

chain.invoke({"question": question})