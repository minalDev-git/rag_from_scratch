# Steps:
# 1. Define a structured object that we want to get out from our llm e.g in this case we wanted one of these ("python-docs", "js-docs", "golang-docs") datasources.
# 2. Take this and convert it into e.g. OpenAI function schema
# 3. Pass that in and bind it to our llm.
# what actually happens is: we ask a question, 
# the llm invokes the function (OpenAI function schema) on the output, to produce an output that adheres the schema that we specified.

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

load_dotenv()
# Setting up a Data Model which is bound to our LLM
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    # Example we have 3 different docs, LLM decides to choose any one and output it
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description= "Given a user question choose which datasource would be most relevant for answering their question."
    )

# LLM with fuction call
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)
# Taking the objct definition, creating the function schema and binding the schema to our llm
structured_llm = llm.with_structured_output(RouteQuery) #produces structured output constrained to these 3 (docs) possibilities

# Prompt
system = """You are an expert in routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

# Define Router
router = prompt | structured_llm

question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human": "speak in {language}"])
prompt.invoke("french")
"""

result = router.invoke({"question": question})

# print(result.datasource) # type: ignore <--- once we have this, we can really easily setup a route

# This function can take the output from the router and do something with it.
# This basically you'll hook the question upto differnt chains
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        # Logic here e.g: apply the question to a retriever full of python information
        return "Chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        # Logic here
        return "Chain for js_docs"
    else:
        # Logic here
        return "Chain for golang_docs"

from langchain_core.runnables import RunnableLambda

full_chain = router | RunnableLambda(choose_route)
full_chain.invoke({"question": question})