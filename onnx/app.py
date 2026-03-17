# onnx_version/app.py
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
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


class ONNXInference:
    def __init__(self, model_path: str, tokenizer):
        self.tokenizer = tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 2
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        
        for input in self.session.get_inputs():
            logger.info(f"  - {input.name}: {input.shape}")
        
        for output in self.session.get_outputs():
            logger.info(f"  - {output.name}: {output.shape}")

    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="np", max_length=128
        )

        outputs = self.session.run(
            ["output"],
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
                "token_type_ids": inputs["token_type_ids"].astype(np.int64),
            },
        )

        cls_embeddings = outputs[0][:, 0, :]

        return cls_embeddings


model = None
onnx_inference = None
model_name = "sergeyzh/rubert-mini-frida"
onnx_model_path = Path("onnx_models/model.onnx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_inference, model

    model = SentenceTransformer(model_name, device="cpu")
    tokenizer = model.tokenizer

    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    onnx_inference = ONNXInference(onnx_model_path, tokenizer)
    
    yield

    del onnx_inference
    del model


app = FastAPI(
    lifespan=lifespan,
)


@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    global onnx_inference

    if onnx_inference is None:
        raise HTTPException(status_code=503, detail="model not installed")

    texts = request.texts if isinstance(request.texts, list) else [request.texts]

    if not texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    try:
        start_time = time.time()

        embeddings = onnx_inference.encode(texts)

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
    uvicorn.run(
        app,
        port=8000,
    )