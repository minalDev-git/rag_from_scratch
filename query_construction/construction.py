import datetime
import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

video_id = "sVcwVQRHIc8&t"
video_url = f"https://www.youtube.com/watch?v={video_id}"
ydl_opts = {'extract_flat': True, 'skip_download': True}

try:
    # 1. Fetch metadata using yt-dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        info = ydl.extract_info(video_url, download=False)
        
        # Format the upload date string from YYYYMMDD to YYYY-MM-DD 00:00:00
        raw_date = info.get("upload_date")
        formatted_date = "Unknown"
        if raw_date:
            formatted_date = datetime.datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d 00:00:00")

        # Map to your exact required metadata structure
        metadata = {
            'source': video_id,
            'title': info.get('title'),
            'description': info.get('description', 'Unknown') or 'Unknown',
            'view_count': info.get('view_count'),
            'thumbnail_url': f"https://i.ytimg.com/vi/{video_id}/hq720.jpg",
            'publish_date': formatted_date,
            'length': info.get('duration'), # yt-dlp uses 'duration' for video length in seconds
            'author': info.get('uploader')  # yt-dlp uses 'uploader' for the channel/author name
        }

        print("=== METADATA ===")
        import pprint
        pprint.pprint(metadata)
except Exception as e:
    print(f"An error occurred: {e}")

# pydantic object (schema of the metadata filters)
class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description= "Similarity search query applied to video transcripts."
    )
    title_search: str = Field(
        ...,
        description= "Alternate version of content search query to apply to video titles." \
        "Should be succint and only include keywords that could be in a video title."
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, insclusive. Only use when explicitly specified."
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use when explicitly specified."
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, insclusive. Only use when explicitly specified."
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use when explicitly specified."
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, insclusive. Only use when explicitly specified."
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, exclusive. Only use when explicitly specified."
    )

    def pretty_print(self) -> None:
        for field_name, field_info in type(self).model_fields.items():
            value = getattr(self, field_name)
            if value is not None and value != field_info.default:
                print(f"{field_name}: {value}")

# Now we prompt the llm to produce queries
load_dotenv()
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
structured_llm = llm.with_structured_output(TutorialSearch)

# Prompt
system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

if there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

# Define Router
query_analyzer = prompt | structured_llm
response = query_analyzer.invoke({"question": "rag_from_scratch"})

# Parse into TutorialSearch so the type checker knows pretty_print exists
result: TutorialSearch = TutorialSearch.model_validate(response)
result.pretty_print()

response = query_analyzer.invoke({"question": "Videos that are focused on the topic of chat langchain that are published before 2024"})
result: TutorialSearch = TutorialSearch.model_validate(response)
result.pretty_print()

response = query_analyzer.invoke({"question": "How to use multi-modal in an agent, only videos under 5 minutes"})
result: TutorialSearch = TutorialSearch.model_validate(response)
result.pretty_print()