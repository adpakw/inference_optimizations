import logging
import time
from contextlib import asynccontextmanager
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    processing_time: float


model = None
device = None
model_name = "sergeyzh/rubert-mini-frida"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device

    device = torch.device("cpu")

    model = SentenceTransformer(model_name, device=device)

    yield

    del model


app = FastAPI(
    lifespan=lifespan,
)


@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="model not installed")

    texts = request.texts if isinstance(request.texts, list) else [request.texts]

    if not texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    try:
        start_time = time.time()

        with torch.no_grad():
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        processing_time = time.time() - start_time

        embeddings_list = embeddings.tolist()

        return EmbeddingResponse(
            embeddings=embeddings_list,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
