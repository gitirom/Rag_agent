import logging
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI

import inngest
import inngest.fast_api

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
    RAGQueryResult,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        # Get the event data safely
        event_data = ctx.event.data

        # Validate pdf_path
        pdf_path = event_data.get("pdf_path")

        if not pdf_path:
            raise ValueError(
                f"Missing 'pdf_path' in event data. "
                f"Received: {event_data}"
            )

        # source_id is optional
        source_id = event_data.get("source_id", pdf_path)

        logging.info(f"Loading PDF: {pdf_path}")

        # Load and chunk PDF
        chunks = load_and_chunk_pdf(pdf_path)

        logging.info(f"Created {len(chunks)} chunks")

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id,
        )

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        logging.info(f"Embedding {len(chunks)} chunks...")

        # Generate embeddings
        vecs = embed_texts(chunks)

        # Generate deterministic IDs
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{i}",
                )
            )
            for i in range(len(chunks))
        ]

        # Create Qdrant payloads
        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
            }
            for i in range(len(chunks))
        ]

        logging.info("Upserting vectors into Qdrant...")

        # Store in Qdrant
        QdrantStorage().upsert(
            ids,
            vecs,
            payloads,
        )

        logging.info(
            f"Successfully inserted {len(chunks)} chunks"
        )

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    # Step 1: Load and chunk PDF
    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    # Step 2: Embed and store in Qdrant
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump()




app = FastAPI()


inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf],
)