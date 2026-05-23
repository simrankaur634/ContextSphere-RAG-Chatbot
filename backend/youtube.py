# RAG using LangChain — Gemini + SentenceTransformers

# 🔧 Install Libraries
# # !pip install -q youtube-transcript-api==0.6.3
# !pip install youtube-transcript-api==0.6.2 -q
# !pip install -q langchain langchain-community langchain-google-genai langchain-core
# !pip install -q faiss-cpu sentence-transformers python-dotenv
# !pip install -q langchain-text-splitters

# 🔑 Set API Key
import os

# Option 1: Colab Secrets (recommended)
try:
    from google.colab import userdata
    os.environ["GOOGLE_API_KEY"] = userdata.get("GEMINI_API_KEY")
    print("✅ API key loaded from Colab Secrets")
except Exception:
    # Option 2: Paste directly
    os.environ["GOOGLE_API_KEY"] = "AIzaSyArgZgU5d_T5Hrki1zu44H9RCsbFYy4CgQ"
    print("⚠️ Using hardcoded API key — use Colab Secrets for production")

# 📦 Imports
# ── Transcript ──────────────────────────────────────────────
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

# ── LangChain core ──────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ── Gemini LLM ──────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI

# ── SentenceTransformer Embeddings ───────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings

print("✅ All imports successful!")

# Nuclear reinstall — run this entire cell if needed
# import subprocess
#
# # Remove completely
# subprocess.run(["pip", "uninstall", "youtube-transcript-api", "-y"])
#
# # Clear any cached versions
# subprocess.run(["pip", "cache", "purge"])
#
# # Reinstall fresh
# result = subprocess.run(
#     ["pip", "install", "youtube-transcript-api==0.6.3", "--no-cache-dir", "--force-reinstall"],
#     capture_output=True, text=True
# )
# print(result.stdout)
# print(result.stderr)

# Step 1a — Fetch YouTube Transcript
video_id = "dDkynerzV-Q"

try:
    # List all available transcripts first
    available = YouTubeTranscriptApi.list_transcripts(video_id)
    print("📋 Available transcripts:")
    for t in available:
        print(f"  - {t.language} ({t.language_code}) | Auto-generated: {t.is_generated}")

    # Try Hindi first, fall back to English
    try:
        transcript_data = available.find_transcript(["en"]).fetch()
        print("\n✅ Hindi transcript downloaded!")
    except Exception:
        transcript_data = available.find_transcript(["en"]).fetch()
        print("\n✅ English transcript downloaded (Hindi not available)!")

    # Handle both old (dict) and new (object) chunk formats
    if hasattr(transcript_data[0], 'text'):
        transcript = " ".join(chunk.text for chunk in transcript_data)
    else:
        transcript = " ".join(chunk["text"] for chunk in transcript_data)

    print(f"📝 Transcript length: {len(transcript)} characters")
    print("\nPreview:", transcript[:300])

except TranscriptsDisabled:
    print("❌ No captions available for this video.")
except Exception as e:
    print(f"❌ Error: {e}")

# Step 1b — Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(f"✅ Total chunks created: {len(chunks)}")
print("\nSample chunk:")
print(chunks[0])

# Step 1c & 1d — Embeddings + Vector Store (SentenceTransformers + FAISS)
# Using SentenceTransformers — runs locally, no API key needed
# 'all-MiniLM-L6-v2' is fast and good for English
# For Hindi/multilingual use: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

print("⏳ Loading embedding model (downloads on first run)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("⏳ Building FAISS vector store...")
vector_store = FAISS.from_documents(chunks, embeddings)

print(f"✅ Vector store ready! Total vectors: {vector_store.index.ntotal}")

# Step 2 — Retrieval
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Test retrieval
test_docs = retriever.invoke("What is DeepMind?")
print(f"✅ Retrieved {len(test_docs)} chunks")
for i, doc in enumerate(test_docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:200])

# Step 3 — Augmentation (Prompt Template)
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
Answer:
""",
    input_variables=["context", "question"]
)

print("✅ Prompt template ready!")

# Step 4 — Generation (Gemini LLM)
# Using Gemini 1.5 Flash — fast and free tier friendly
# Alternatives: 'gemini-1.5-pro', 'gemini-2.0-flash'
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

print("✅ Gemini LLM ready!")

# Quick manual test
question = "Is nuclear fusion discussed in this video? If yes, what was discussed?"
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = llm.invoke(final_prompt)
print("\n🤖 Answer:")
print(answer.content)

# 🔗 Full RAG Chain
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print("✅ Full RAG chain assembled!")

# 🚀 Ask anything about the video!
response = main_chain.invoke("Can you summarize the video?")
print("🤖 Answer:")
print(response)

# Try more questions
questions = [
    "Who is Demis Hassabis?",
    "What is DeepMind working on?",
    "Is nuclear fusion discussed?"
]

for q in questions:
    print(f"\n❓ {q}")
    print("🤖", main_chain.invoke(q))
    print("-" * 60)
