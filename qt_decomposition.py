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
import langchainhub as hub

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

# Multi-Query: Different Perspectives
template = """
You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems/ sub-questions that can be answered in isolation.
Output(3 queries): {question}
"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = (
    prompt_decomposition 
    | llm 
    | StrOutputParser()
    | (lambda x: x.split('\n'))
)

# Retrieve
question = "What are the main components of Langraph that allow parallel-execution?"
questions = generate_queries_decomposition.invoke({"question":question})

# IR-COT RAG
# template = """Here is the question you eed to answer:
# \n --- \n {question} \n --- \n
# Here is any available background question + answer pairs:
# \n --- \n {q_a_pairs} \n --- \n
# Here is additional context relevant to the question:
# \n --- \n {context} \n --- \n
# Use the above context and any background question + answer pairs to answer the question: \n {question}
# """
# decomposition_prompt = ChatPromptTemplate.from_template(template)

# def format_qa_pair(question, answer):
#     """Format Q and A pair"""
#     formatted_string = ""
#     formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
#     return formatted_string

# q_a_pairs = ""
# for q in questions:

#     rag_chain = (
#         {"context": itemgetter("question") | retriever, 
#         "question": itemgetter("question"),
#         "q_a_pairs": itemgetter("q_a_pairs")
#         }
#         | decomposition_prompt 
#         | llm 
#         | StrOutputParser()
#     )
#     answer = rag_chain.invoke({"question":q, "q_a_pairs": q_a_pairs})
#     print(f"q: {q}, Ans: {answer} \n")
#     q_a_pair = format_qa_pair(q,answer)
#     q_a_pairs = q_a_pairs + "\n --- \n" + q_a_pair

# Least-most RAG

client = hub.Client()
prompt_rag = client.pull("rlm/rag-prompt")

def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """Rag on each Sub-Question"""

    # Use our decomposition
    sub_questions = sub_question_generator_chain.invoke({"question":question})

    # Initialize a list to hold RAG chain results
    rag_results = []

    for sub_question in sub_questions:

        # Retrieves documents for each sub-question
        retrieved_docs = retriever.invoke(sub_question)

        # Use retrieved documents and the sub_question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, "question": sub_question})

        rag_results.append(answer)

    return rag_results, sub_questions

# Wrap the retrieval and RAG process in a RunnableLamda for integration in a chain.
answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries_decomposition)

def format_qa_pair(questions, answers):
    """Format Q and A pair"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

template = """Here is a set of Q+A pairs:
{context}
Use this to synthesize an answer to the question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt 
    | llm 
    | StrOutputParser()
)
response = final_rag_chain.invoke({"question":question})
print(response)