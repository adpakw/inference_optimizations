import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    request_id: Optional[str] = None


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    processing_time: float
    request_id: Optional[str] = None
    batched: bool = False


@dataclass
class QueueItem:
    texts: List[str]
    future: asyncio.Future
    request_id: str
    timestamp: float


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

    def encode_batch(self, texts_batch: List[List[str]]) -> List[np.ndarray]:
        all_texts = []
        request_lengths = []

        for texts in texts_batch:
            request_lengths.append(len(texts))
            all_texts.extend(texts)

        inputs = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=128,
        )

        outputs = self.session.run(
            ["output"],
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
                "token_type_ids": inputs["token_type_ids"].astype(np.int64),
            },
        )

        all_embeddings = outputs[0][:, 0, :]

        result_embeddings = []
        start_idx = 0
        for length in request_lengths:
            result_embeddings.append(all_embeddings[start_idx : start_idx + length])
            start_idx += length

        return result_embeddings


class BatchProcessor:
    def __init__(
        self, onnx_inference, max_batch_size: int = 32, max_wait_time: float = 1
    ):
        self.onnx_inference = onnx_inference
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = True

        logger.info(
            f"BatchProcessor initialized with max_batch_size={max_batch_size}, max_wait_time={max_wait_time}s"
        )

    async def process_requests(self):
        while self.is_running:
            try:
                try:
                    first_item = await asyncio.wait_for(
                        self.queue.get(), timeout=self.max_wait_time
                    )
                except asyncio.TimeoutError:
                    continue

                batch_items = [first_item]
                batch_start_time = time.time()

                while len(batch_items) < self.max_batch_size:
                    try:
                        remaining_time = self.max_wait_time - (
                            time.time() - batch_start_time
                        )
                        if remaining_time <= 0:
                            break

                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=remaining_time
                        )
                        batch_items.append(item)
                    except asyncio.TimeoutError:
                        break

                await self._process_batch(batch_items)

            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                for item in batch_items if "batch_items" in locals() else []:
                    if not item.future.done():
                        item.future.set_exception(e)

    async def _process_batch(self, batch_items: List[QueueItem]):
        start_time = time.time()
        batch_size = len(batch_items)
        total_texts = sum(len(item.texts) for item in batch_items)

        logger.info(
            f"Processing batch of {batch_size} requests ({total_texts} texts total)"
        )

        try:
            texts_batch = [item.texts for item in batch_items]

            embeddings_batch = await asyncio.get_event_loop().run_in_executor(
                None, self.onnx_inference.encode_batch, texts_batch
            )

            processing_time = time.time() - start_time

            for item, embeddings in zip(batch_items, embeddings_batch):
                if not item.future.done():
                    item.future.set_result(
                        {
                            "embeddings": embeddings.tolist(),
                            "processing_time": processing_time,
                            "request_id": item.request_id,
                            "batched": True,
                        }
                    )

            logger.info(
                f"Batch processed in {processing_time:.3f}s, avg per text: {processing_time/total_texts:.3f}s"
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for item in batch_items:
                if not item.future.done():
                    item.future.set_exception(e)

    async def add_request(self, texts: List[str], request_id: str) -> Dict[str, Any]:
        future = asyncio.Future()
        item = QueueItem(
            texts=texts, future=future, request_id=request_id, timestamp=time.time()
        )

        await self.queue.put(item)

        try:
            result = await future
            return result
        except Exception as e:
            raise e

    async def start(self):
        self.processing_task = asyncio.create_task(self.process_requests())

    async def stop(self):
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass


model = None
onnx_inference = None
batch_processor = None
model_name = "sergeyzh/rubert-mini-frida"
onnx_model_path = Path("onnx_models/model.onnx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_inference, model, batch_processor

    model = SentenceTransformer(model_name, device="cpu")
    tokenizer = model.tokenizer

    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    onnx_inference = ONNXInference(onnx_model_path, tokenizer)

    batch_processor = BatchProcessor(onnx_inference, max_batch_size=32, max_wait_time=0.5)
    await batch_processor.start()

    yield

    await batch_processor.stop()
    del onnx_inference
    del model


app = FastAPI(
    lifespan=lifespan,
)


@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    global batch_processor

    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    texts = request.texts if isinstance(request.texts, list) else [request.texts]

    if not texts:
        raise HTTPException(status_code=400, detail="Texts list is empty")

    request_id = request.request_id or str(uuid.uuid4())

    try:
        result = await batch_processor.add_request(texts, request_id)

        return EmbeddingResponse(
            embeddings=result["embeddings"],
            processing_time=result["processing_time"],
            request_id=result["request_id"],
            batched=result["batched"],
        )

    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        port=8000,
    )
