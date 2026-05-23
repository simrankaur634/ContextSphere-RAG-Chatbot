from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    full_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_pdf_by_page(file_path: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    reader = PdfReader(file_path)
    chunks = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        text = page.extract_text()
        if not text:
            continue
        
        # Clean text spacing
        text = " ".join(text.split())

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                # Prepend the page annotation prefix so RAG prompts inherit page context
                chunks.append(f"[Page {page_num}] {chunk}")
            start += chunk_size - overlap

    return chunks