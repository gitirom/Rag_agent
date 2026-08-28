import logging
import uuid
import datetime
import os

from dotenv import load_dotenv
from fastapi import FastAPI

import inngest
import inngest.fast_api
from inngest.experimental import ai

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



@inngest_client.create_function(
    fn_id="RAG:  Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)

#top_k = 5: referce to 5 most relevant chunks.
    # Example of a query and its results:
    #         Chunk 124 → similarity 0.92
    #         Chunk 87  → similarity 0.89
    #         Chunk 421 → similarity 0.86
    #         Chunk 52  → similarity 0.83
    #         Chunk 901 → similarity 0.81
    #  

async def rag_query_pdf_ai(ctx: inngest.Context):


    question = ctx.event.data.get("question")

    if not question:
        raise ValueError(
            "Missing 'question' in event data."
        )

    # Number of chunks to retrieve
    top_k = ctx.event.data.get("top_k", 5)

    # Make sure top_k is reasonable
    top_k = max(1, min(top_k, 20))


    def search_documents() -> RAGSearchResult:

        query_vector = embed_texts(
            [question]
        )[0]

        store = QdrantStorage()

        found = store.search(
            query_vector,
            top_k=top_k,
        )

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
        )


    found = await ctx.step.run(
        "embed-and-search",
        search_documents,
        output_type=RAGSearchResult,
    )



    if not found.contexts:

        return {
            "answer": (
                "I couldn't find relevant information "
                "in the provided documents."
            ),
            "sources": [],
            "num_contexts": 0,
        }



    # 4. Build context


    context_block = "\n\n".join(
        f"- {chunk}"
        for chunk in found.contexts
    )


    # 5. Build LLM prompt


    user_content = f"""
            Use the following context to answer the question.

            Context:
            {context_block}

            Question:
            {question}

            Instructions:
            - Answer using only the provided context.
            - Do not use outside knowledge.
            - If the answer is not contained in the context,
            say: "I don't have enough information in the provided documents."
            - Keep the answer concise.
        """



    openrouter_api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured."
        )


    adapter = ai.openai.Adapter(
        auth_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model="minimax/minimax-m3:free",
    )



    res = await ctx.step.ai.infer(
        "llm-answer",

        adapter=adapter,

        body={
            "max_tokens": 1024,

            "temperature": 0.2,

            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a RAG assistant. "
                        "Answer questions using only "
                        "the provided context."
                    ),
                },

                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        },
    )


    # 8. Extract LLM answer


    answer = (
        res["choices"][0]
        ["message"]
        ["content"]
        .strip()
    )


    # 9. Return final RAG result


    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }

app = FastAPI()


inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai],
)





