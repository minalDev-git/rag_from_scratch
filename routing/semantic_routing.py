import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Literal
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()
# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a consise and easy to understand manner.\
When you don't know the answer to a question, you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# embedded prompts
prompt_templates = [physics_template, math_template]
prompt_embeddings = embedding_model.embed_documents(prompt_templates)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

# Route question to prompt
def prompt_router(input):
    # Enbed question
    query_embedding = embedding_model.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding],prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")

    return PromptTemplate.from_template(most_similar)

chain = (
    RunnablePassthrough()
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

print(chain.invoke({"query": "What is a black hole?"}))