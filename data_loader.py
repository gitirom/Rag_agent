import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()


api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set")

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

# ============================================================
# Embedding Configuration
# ============================================================

EMBED_MODEL = "liquid/lfm-2.5-embedding-350m:free"
EMBED_DIM = 1024


# ============================================================
# Text Chunking
# ============================================================

# The OpenRouter model has a 512-token input limit. Keep chunks below that limit.
splitter = SentenceSplitter(
    chunk_size=400,
    chunk_overlap=50,
)


# ============================================================
# PDF → Chunks
# ============================================================

def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF and split its text into smaller chunks.
    """

    docs = PDFReader().load_data(file=path)

    texts = [
        doc.text
        for doc in docs
        if getattr(doc, "text", None)
    ]

    chunks = []

    for text in texts:
        chunks.extend(
            splitter.split_text(text)
        )

    return chunks


# ============================================================
# Text → Embeddings
# ============================================================

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks.
    """

    if not texts:
        return []

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    return embeddings


# ============================================================
# Query → Embedding
# ============================================================

# def embed_query(query: str) -> list[float]:
#     """ 
#     Generate an embedding for a user query. for docs doc embedding and for qeurys use query this is the model requirement. 
#     """

#     if not query.strip():
#         raise ValueError("Query cannot be empty.")

#     response = client.embeddings.create(
#         model=EMBED_MODEL,
#         input=[query],
#     )

#     return response.data[0].embedding



