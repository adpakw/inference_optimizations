import logging
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "sergeyzh/rubert-mini-frida"
output_dir = Path("onnx_models")
output_dir.mkdir(parents=True, exist_ok=True)


def convert_to_onnx():
    model = SentenceTransformer(model_name, device="cpu")

    transformer_model = model[0].auto_model
    tokenizer = model.tokenizer

    dummy_text = ["dummy_text"]
    inputs = tokenizer(
        dummy_text, padding=True, truncation=True, return_tensors="pt", max_length=512
    )

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "token_type_ids": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    }

    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        transformer_model,
        (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    return onnx_path


if __name__ == "__main__":
    convert_to_onnx()
