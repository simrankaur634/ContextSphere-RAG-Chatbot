import os
import shutil

from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import SessionLocal, engine
import models
from pdf_utils import extract_text_from_pdf, chunk_text, chunk_pdf_by_page
from csv_utils import extract_data_from_csv
from rag_pipeline import retrieve_relevant_chunks, build_rag_prompt, ask_ai, web_search
from youtube_summarizer import process_and_summarize_youtube, clean_video_id

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    user_id: int
    title: str


class MessageRequest(BaseModel):
    chat_id: int
    message: str
    web_search_enabled: bool = False
    model_provider: str = "mistral"


@app.get("/")
def home():
    from fastapi.responses import FileResponse
    return FileResponse("../frontend/index.html")

from fastapi.staticfiles import StaticFiles
app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")


@app.post("/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == data.email).first()
    if existing:
        return {"error": "User already exists"}

    user = models.User(name=data.name, email=data.email, password=data.password)
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"id": user.id, "name": user.name, "email": user.email}


@app.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(models.User).filter(models.User.email == data.email).first()

        if not user:
            return {"error": "User not found"}

        if user.password != data.password:
            return {"error": "Invalid password"}

        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        }

    except Exception as e:
        print("LOGIN ERROR:", str(e))
        return {"error": str(e)}


@app.post("/create_chat")
def create_chat(data: ChatRequest, db: Session = Depends(get_db)):
    chat = models.Chat(user_id=data.user_id, title=data.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"id": chat.id, "title": chat.title, "user_id": chat.user_id}


@app.get("/get_chats/{user_id}")
def get_chats(user_id: int, db: Session = Depends(get_db)):
    chats = db.query(models.Chat).filter(models.Chat.user_id == user_id).all()
    return [{"id": c.id, "title": c.title, "user_id": c.user_id} for c in chats]


@app.get("/get_messages/{chat_id}")
def get_messages(chat_id: int, db: Session = Depends(get_db)):
    messages = db.query(models.Message).filter(models.Message.chat_id == chat_id).all()
    return [
        {"id": m.id, "role": m.role, "content": m.content, "chat_id": m.chat_id}
        for m in messages
    ]


@app.get("/get_documents/{user_id}")
def get_documents(user_id: int, db: Session = Depends(get_db)):
    docs = db.query(models.Document).filter(models.Document.user_id == user_id).all()
    return [{"id": d.id, "filename": d.filename, "chat_id": d.chat_id} for d in docs]


@app.post("/upload_file")
def upload_file(
    chat_id: int = Form(...),
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    ext = file.filename.lower().split('.')[-1]
    if ext not in ["pdf", "csv"]:
        return {"error": "Only PDF and CSV files are allowed."}

    saved_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if ext == "pdf":
        chunks = chunk_pdf_by_page(saved_path, chunk_size=800, overlap=150)
        if not chunks:
            return {"error": "No readable content found in PDF."}
    else: # csv
        extracted_text = extract_data_from_csv(saved_path)
        if not extracted_text.strip():
            return {"error": "No readable content found in CSV."}
        # For CSV, store summary as a single chunk if it fits
        chunks = [extracted_text] if len(extracted_text) < 5000 else chunk_text(extracted_text, chunk_size=2000, overlap=200)

    document = models.Document(
        user_id=user_id,
        chat_id=chat_id,
        filename=file.filename
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    for i, chunk in enumerate(chunks):
        db_chunk = models.DocumentChunk(
            document_id=document.id,
            chat_id=chat_id,
            chunk_index=i,
            content=chunk
        )
        db.add(db_chunk)

    db.commit()

    return {
        "message": f"{ext.upper()} uploaded and processed successfully.",
        "document_id": document.id,
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }


@app.post("/send_message")
def send_message(data: MessageRequest, db: Session = Depends(get_db)):
    user_msg = models.Message(
        chat_id=data.chat_id,
        role="user",
        content=data.message
    )
    db.add(user_msg)
    db.commit()

    history_rows = db.query(models.Message).filter(
        models.Message.chat_id == data.chat_id
    ).all()

    # Build a strictly alternating, error-free chat history list
    history = []
    for m in history_rows:
        content = m.content
        # Skip saving API errors in our compiled history
        if content.startswith("Error:") or content.startswith("API Error") or content.startswith("API Connection Error") or content.startswith("All OpenRouter"):
            continue
        
        # Enforce alternating roles (user -> assistant -> user).
        # If we have two consecutive user messages (due to a previous failed turn),
        # keep only the most recent user query to keep the LLM context perfectly clean.
        if history and history[-1]["role"] == m.role:
            if m.role == "user":
                history[-1]["content"] = content
            continue
            
        history.append({"role": m.role, "content": content})

    chunk_rows = db.query(models.DocumentChunk).filter(
        models.DocumentChunk.chat_id == data.chat_id
    ).order_by(models.DocumentChunk.chunk_index).all()

    chunks = [{"content": c.content} for c in chunk_rows]

    # ---------- Image Generation Shortcut ----------
    # Detect commands that request image creation. Supports:
    #   /image <prompt>
    #   generate an image of <prompt>
    #   create an image of <prompt>
    #   draw an image of <prompt>
    img_cmd_prefixes = ["/image ", "generate an image of ", "create an image of ", "draw an image of "]
    lowered_msg = data.message.lower().strip()
    is_image_request = False
    img_prompt = ""
    for prefix in img_cmd_prefixes:
        if lowered_msg.startswith(prefix):
            is_image_request = True
            img_prompt = data.message[len(prefix):].strip()
            break
    # Also catch phrases anywhere in the sentence
    if not is_image_request:
        for phrase in ["generate an image of", "create an image of", "draw an image of"]:
            if phrase in lowered_msg:
                is_image_request = True
                img_prompt = data.message.lower().split(phrase, 1)[1].strip()
                break

    if is_image_request and img_prompt:
        # Optional: enhance the prompt using LLM for richer images
        try:
            enhanced_prompt = ask_ai(f"Rewrite this description to be a detailed, high‑quality image prompt for an AI art model: {img_prompt}", model_provider=data.model_provider)
        except Exception:
            enhanced_prompt = img_prompt

        # Encode prompt for URL
        from urllib.parse import quote
        encoded = quote(enhanced_prompt)
        pollinations_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&private=true"
        try:
            import requests
            img_resp = requests.get(pollinations_url, timeout=30)
            img_resp.raise_for_status()
        except Exception as e:
            # Return a graceful error message without polluting DB
            return {"response": f"*Failed to generate image: {str(e)}*"}

        # Save image locally
        import uuid, datetime
        filename = f"generated_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}.jpg"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(img_resp.content)

        # Register the image as a document (so it appears in the document list)
        doc = models.Document(user_id=None, chat_id=data.chat_id, filename=filename)
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Respond with markdown image
        image_url = f"http://127.0.0.1:8000/uploads/{filename}"
        ai_response = f"![Generated Image]({image_url})"
        # Skip the normal RAG flow – directly send the image response
        # Save assistant message and return
        ai_msg = models.Message(chat_id=data.chat_id, role="assistant", content=ai_response)
        db.add(ai_msg)
        db.commit()
        return {"response": ai_response}
    # ----------------------------------------------------

    # Check for summary or analysis request or direct Q&A
    useful_chunks = []
    if chunks:
        msg_lower = data.message.lower()
        is_summary = "summarise" in msg_lower or "summarize" in msg_lower
        is_analysis = "analyse" in msg_lower or "analyze" in msg_lower or "chart" in msg_lower or "graph" in msg_lower
        
        if (is_summary or is_analysis) and ("pdf" in msg_lower or "csv" in msg_lower or "data" in msg_lower):
            print("Extended context request detected...")
            # If a specific filename is mentioned, try to filter chunks for that document
            docs = db.query(models.Document).filter(models.Document.chat_id == data.chat_id).all()
            target_chunks = chunks
            for doc in docs:
                if doc.filename.lower() in msg_lower:
                    doc_chunk_rows = db.query(models.DocumentChunk).filter(
                        models.DocumentChunk.document_id == doc.id
                    ).order_by(models.DocumentChunk.chunk_index).all()
                    target_chunks = [{"content": c.content} for c in doc_chunk_rows]
                    break
            
            useful_chunks = target_chunks[:80] 
        else:
            relevant_chunks = retrieve_relevant_chunks(data.message, chunks, top_k=4)
            useful_chunks = [c for c in relevant_chunks if c.get("score", 0) > 0]

    web_results = []
    if data.web_search_enabled:
        web_results = web_search(data.message)

    prompt = build_rag_prompt(
        question=data.message,
        relevant_chunks=useful_chunks,
        history=history,
        web_results=web_results
    )
    ai_response = ask_ai(prompt, model_provider=data.model_provider)

    # Determine if the response was an API or network error
    is_error = (
        ai_response.startswith("Error:") or 
        ai_response.startswith("API Error") or 
        ai_response.startswith("API Connection Error") or 
        ai_response.startswith("All OpenRouter")
    )

    if is_error:
        # DO NOT save the error message, and delete the failed user message from database to keep history pristine
        db.delete(user_msg)
        db.commit()
    else:
        # Save a valid AI assistant message to the database
        ai_msg = models.Message(
            chat_id=data.chat_id,
            role="assistant",
            content=ai_response
        )
        db.add(ai_msg)
        db.commit()

        # AUTO RENAME CHAT
        chat = db.query(models.Chat).filter(models.Chat.id == data.chat_id).first()
        if chat and chat.title == "New Chat":
            doc = db.query(models.Document).filter(models.Document.chat_id == data.chat_id).first()
            doc_name = doc.filename if doc else "Chat"
            short_question = " ".join(data.message.split()[:5])
            new_title = f"{doc_name} - {short_question}"
            chat.title = new_title[:50]
            db.commit()

    return {"response": ai_response}


@app.delete("/delete_chats/{user_id}")
def delete_chats(user_id: int, db: Session = Depends(get_db)):
    chats = db.query(models.Chat).filter(models.Chat.user_id == user_id).all()
    for chat in chats:
        db.query(models.Message).filter(models.Message.chat_id == chat.id).delete()
        db.query(models.DocumentChunk).filter(models.DocumentChunk.chat_id == chat.id).delete()
        db.query(models.Document).filter(models.Document.chat_id == chat.id).delete()
        db.delete(chat)
    db.commit()
    return {"status": "All chats deleted successfully"}


class RenameChatRequest(BaseModel):
    chat_id: int
    new_title: str


@app.put("/rename_chat")
def rename_chat(data: RenameChatRequest, db: Session = Depends(get_db)):
    chat = db.query(models.Chat).filter(models.Chat.id == data.chat_id).first()
    if not chat:
        return {"error": "Chat not found"}
    chat.title = data.new_title
    db.commit()
    return {"message": "Chat renamed successfully"}


class SummarizeNewsRequest(BaseModel):
    domain: str
    articles: list


class YoutubeSummarizeRequest(BaseModel):
    video_ids: list
    chat_id: int
    user_id: int


@app.post("/summarize_youtube")
def summarize_youtube(data: YoutubeSummarizeRequest, db: Session = Depends(get_db)):
    """Process a list of YouTube video IDs: fetch transcripts, generate summaries, and store chunks."""
    if not data.video_ids:
        return {"error": "No video IDs provided."}

    results = []
    errors = []

    for raw_id in data.video_ids:
        vid = clean_video_id(str(raw_id).strip())
        if not vid:
            errors.append({"video_id": raw_id, "error": "Invalid video ID or URL."})
            continue
        try:
            result = process_and_summarize_youtube(
                video_id=vid,
                chat_id=data.chat_id,
                user_id=data.user_id,
                db=db
            )
            results.append(result)
        except Exception as e:
            errors.append({"video_id": vid, "error": str(e)})

    return {"results": results, "errors": errors}


@app.get("/get_news")
def get_news(domain: str):
    domain_lower = domain.lower()
    queries = {
        "finance": "latest financial market business news and stock market trends",
        "sports": "latest global sports headlines match scores tournament news",
        "agriculture": "latest agricultural technology farming crop yield and agro business news",
        "education": "latest academic learning educational research university edtech news",
        "fashion": "latest fashion designer style trends beauty and runway show news"
    }
    query = queries.get(domain_lower, f"latest {domain_lower} news")
    results = web_search(query, max_results=8)
    return {"domain": domain, "results": results}


@app.post("/summarize_news")
def summarize_news(data: SummarizeNewsRequest):
    if not data.articles:
        return {"summary": "No news articles found to summarize."}
        
    articles_context = ""
    for idx, art in enumerate(data.articles):
        articles_context += f"Article #{idx+1}:\nTitle: {art.get('title', '')}\nSnippet: {art.get('body', '')}\nSource: {art.get('href', '')}\n\n"
        
    prompt = f"""
You are AnswerAI, an intelligent assistant.
Summarize the following latest news articles for the '{data.domain.upper()}' domain into a highly professional, cohesive, and premium news digest briefing.

Key requirements:
1. Provide an executive summary of the overall current trend in this domain.
2. Group the major highlights into distinct categories or bullet points with clear, bold headers.
3. Keep the tone insightful, informative, and completely free of meta-talk.
4. Keep the summary detailed but highly readable.

Articles Context:
{articles_context}

Provide a well-structured summary:
"""
    try:
        summary = ask_ai(prompt)
    except Exception as e:
        summary = f"Could not generate summary due to an error: {str(e)}"
    return {"summary": summary}


@app.delete("/delete_chat/{chat_id}")
def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    # 1. Delete associated messages
    db.query(models.Message).filter(models.Message.chat_id == chat_id).delete()
    
    # 2. Find and delete documents and chunks
    docs = db.query(models.Document).filter(models.Document.chat_id == chat_id).all()
    for doc in docs:
        db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == doc.id).delete()
        file_path = os.path.join("uploads", doc.filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print("Error removing file:", e)
        db.delete(doc)
        
    # 3. Delete the chat itself
    chat = db.query(models.Chat).filter(models.Chat.id == chat_id).first()
    if chat:
        db.delete(chat)
        db.commit()
        return {"message": "Chat and all associated messages/documents deleted successfully."}
    
    return {"error": "Chat not found."}


@app.delete("/delete_document/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        return {"error": "Document not found."}
        
    db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == document_id).delete()
    
    file_path = os.path.join("uploads", doc.filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print("Error removing file:", e)
            
    db.delete(doc)
    db.commit()
    return {"message": "Document and chunks deleted successfully."}


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")