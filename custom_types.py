import pydantic

"""
defining data structures: typed, validated objects = this makes the code easier to understand and maintain, and helps catch errors early in the development process. 
"""


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

    """
    example of returned data from the ingestion process:
    [
    "Artificial intelligence is...",
    "Machine learning is a subset of AI...",
    "Neural networks are..."
    ]
    document_123.pdf
    """


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int