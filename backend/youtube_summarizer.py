import os
import sys
import re
import requests
from dotenv import load_dotenv

# Import transcript libraries
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

# Import LangChain libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

def get_google_api_key():
    """Retrieve Google API Key from environment or fallback."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Fallback to the key used in youtube.py if not in env
        api_key = "AIzaSyArgZgU5d_T5Hrki1zu44H9RCsbFYy4CgQ"
    return api_key

def get_youtube_title(video_id: str) -> str:
    """Fetch the title of a YouTube video without using the YouTube API."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            match = re.search(r"<title>(.*?)</title>", response.text)
            if match:
                title = match.group(1).replace(" - YouTube", "").strip()
                return title
    except Exception as e:
        print(f"Warning: Could not retrieve video title for {video_id}: {e}")
    return f"YouTube Video {video_id}"

def fetch_transcript(video_id: str) -> str:
    """Fetch the transcript of the YouTube video, prioritizing Hindi and English."""
    try:
        available = YouTubeTranscriptApi().list(video_id)
        
        # Try to find English ('en') or Hindi ('hi') transcripts
        try:
            transcript_data = available.find_transcript(["en", "hi"]).fetch()
        except Exception:
            # Fallback to the first available transcript (any language)
            transcript_data = next(iter(available)).fetch()

        if hasattr(transcript_data[0], 'text'):
            transcript = " ".join(chunk.text for chunk in transcript_data)
        else:
            transcript = " ".join(chunk["text"] for chunk in transcript_data)
            
        return transcript
    except TranscriptsDisabled:
        raise Exception("Captions/transcripts are disabled for this YouTube video.")
    except Exception as e:
        raise Exception(f"Failed to fetch YouTube transcript: {str(e)}")

def generate_summary(transcript: str, video_title: str) -> str:
    """Generate a comprehensive summary of the transcript using Gemini or fallback ask_ai."""
    prompt_text = f"""
You are AnswerAI, a premium content analysis assistant. 
Analyze the provided transcript for the video titled "{video_title}".

Create a structured, detailed, and professional news-briefing style summary of the video.
Use the following structure:
1. **Executive Summary**: A concise overview of the main topic and purpose of the video (3-4 sentences).
2. **Key Themes & Topics**: Identify the main topics discussed, with bold headings and structured bullet points highlighting the core concepts.
3. **Important Details & Takeaways**: Summarize any notable quotes, examples, or specific stats shared in the video.
4. **Conclusion**: A final closing thought on the video's significance.

Keep the tone insightful, clear, and professional. Avoid meta-talk like "In this transcript..." or "The video says...".

Transcript:
{transcript[:800000]}

Premium Summary:
"""

    api_key = get_google_api_key()
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.25,
                google_api_key=api_key
            )
            prompt = PromptTemplate(
                template="""
You are AnswerAI, a premium content analysis assistant. 
Analyze the provided transcript for the video titled "{title}".

Create a structured, detailed, and professional news-briefing style summary of the video.
Use the following structure:
1. **Executive Summary**: A concise overview of the main topic and purpose of the video (3-4 sentences).
2. **Key Themes & Topics**: Identify the main topics discussed, with bold headings and structured bullet points highlighting the core concepts.
3. **Important Details & Takeaways**: Summarize any notable quotes, examples, or specific stats shared in the video.
4. **Conclusion**: A final closing thought on the video's significance.

Keep the tone insightful, clear, and professional. Avoid meta-talk like "In this transcript..." or "The video says...".

Transcript:
{context}

Premium Summary:
""",
                input_variables=["context", "title"]
            )
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"context": transcript[:800000], "title": video_title})
        except Exception as e:
            print(f"Warning: Gemini API call failed ({e}). Falling back to main AI model...")
            
    # Fallback to the project's main ask_ai function (Mistral/OpenRouter)
    try:
        from rag_pipeline import ask_ai
        return ask_ai(prompt_text)
    except Exception as err:
        raise Exception(f"Failed to generate summary: {err}")

def save_to_database(video_id: str, video_title: str, transcript: str, chat_id: int, user_id: int, db) -> int:
    """Save the YouTube video as a document and its chunks as document chunks in the database."""
    import models
    
    # Create Document record
    doc_filename = f"YouTube - {video_title[:60]}"
    if len(video_title) > 60:
        doc_filename += "..."
    doc_filename += f" ({video_id})"
    
    # First, check if this video has already been uploaded for this chat to prevent duplicates
    existing_doc = db.query(models.Document).filter(
        models.Document.chat_id == chat_id,
        models.Document.filename == doc_filename
    ).first()
    
    if existing_doc:
        # Delete old chunks and document first to refresh/update
        db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == existing_doc.id).delete()
        db.delete(existing_doc)
        db.commit()

    db_document = models.Document(
        user_id=user_id,
        chat_id=chat_id,
        filename=doc_filename
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    # Chunk the transcript using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.create_documents([transcript])
    
    # Insert chunks to database
    for i, chunk in enumerate(chunks):
        db_chunk = models.DocumentChunk(
            document_id=db_document.id,
            chat_id=chat_id,
            chunk_index=i,
            content=f"[YouTube Context] {chunk.page_content}"
        )
        db.add(db_chunk)
        
    db.commit()
    return db_document.id

def process_and_summarize_youtube(video_id: str, chat_id: int, user_id: int, db) -> dict:
    """Fetch transcript, generate summary, and save chunks to the database."""
    # Clean the video_id in case a full URL was provided
    video_id = clean_video_id(video_id)
    
    title = get_youtube_title(video_id)
    transcript = fetch_transcript(video_id)
    summary = generate_summary(transcript, title)
    doc_id = save_to_database(video_id, title, transcript, chat_id, user_id, db)
    
    return {
        "video_id": video_id,
        "title": title,
        "summary": summary,
        "document_id": doc_id,
        "transcript_length": len(transcript)
    }

def clean_video_id(video_id_or_url: str) -> str:
    """Extract 11-character YouTube video ID from a potential URL or string."""
    video_id_or_url = video_id_or_url.strip()
    
    # Match standard youtube watch link, shared link, embed link, or shorts link
    regexes = [
        r"(?:v=|\/v\/|embed\/|youtu\.be\/|shorts\/|\/e\/|watch\?v=)([^#\&\?]*)"
    ]
    
    for regex in regexes:
        match = re.search(regex, video_id_or_url)
        if match:
            extracted_id = match.group(1)
            if len(extracted_id) == 11:
                return extracted_id
                
    # If it's already an 11-char string, return it
    if len(video_id_or_url) == 11:
        return video_id_or_url
        
    return video_id_or_url

if __name__ == "__main__":
    # Command Line Interface execution
    print("=== YouTube Transcript & Summarizer CLI ===")
    
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python youtube_summarizer.py <video_id_or_url_1> <video_id_or_url_2> ...")
        sys.exit(1)
        
    ids_to_process = sys.argv[1:]
    
    for idx, raw_id in enumerate(ids_to_process):
        v_id = clean_video_id(raw_id)
        print(f"\n[{idx+1}/{len(ids_to_process)}] Processing Video ID: {v_id} ...")
        
        try:
            print("Fetching video title...")
            title = get_youtube_title(v_id)
            print(f"Title: {title}")
            
            print("Retrieving transcript...")
            transcript = fetch_transcript(v_id)
            print(f"Transcript loaded ({len(transcript)} chars).")
            
            print("Generating summary using Gemini 1.5 Flash...")
            summary = generate_summary(transcript, title)
            
            print("\n" + "="*40)
            print(f"Summary for: {title}")
            print("="*40)
            safe_summary = summary.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
            print(safe_summary)
            print("="*40 + "\n")
            
        except Exception as e:
            print(f"Error processing {raw_id}: {e}")
