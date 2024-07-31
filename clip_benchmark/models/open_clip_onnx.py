import os
from app.models.clip.textual import MClipTextualEncoder, OpenClipTextualEncoder
from app.models.clip.visual import OpenClipVisualEncoder
from app.models.constants import get_model_source
from app.schemas import ModelSource

import numpy as np
from tokenizers import Encoding
import torch


class OpenCLIPEncoder:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = os.environ.get("CACHE_DIR"),
    ):
        model_name = model_name.replace("xlm-roberta-base", "XLM-Roberta-Base")
        model_name = model_name.replace("xlm-roberta-large", "XLM-Roberta-Large")
        source = get_model_source(model_name)
        match source:
            case ModelSource.MCLIP:
                self.textual = MClipTextualEncoder(model_name, cache_dir, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            case ModelSource.OPENCLIP:
                self.textual = OpenClipTextualEncoder(model_name, cache_dir, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            case _:
                raise ValueError(f"Unknown model: {model_name}")

        self.visual = OpenClipVisualEncoder(model_name, cache_dir, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.visual.download()
        self.visual._load_transform()
        self.textual.tokenizer = self.textual._load_tokenizer()

    def encode_text(self, tokens: str):
        self.textual.load()
        return torch.from_numpy(self.textual.session.run(None, tokens)[0])

    def encode_image(self, images: dict[str, torch.Tensor]):
        self.visual.load()
        images["image"] = np.squeeze(images["image"].numpy(), 1)
        return torch.from_numpy(self.visual.session.run(None, images)[0])

    def eval(self): ...


def load_open_clip_onnx(
    model_name: str,
    pretrained: str | None = None,
    cache_dir: str | None = None,
    device="cpu",
):
    model = OpenCLIPEncoder(
        model_name=f"{model_name}__{pretrained}" if pretrained else model_name,
        cache_dir=cache_dir,
    )

    def tokenize(text: list[str]):
        out: list[Encoding] = model.textual.tokenizer.encode_batch(text)
        return {
            "attention_mask": np.array(
                [b.attention_mask for b in out], dtype=np.int32
            ),
            "input_ids": np.array([b.ids for b in out], dtype=np.int32),
        }

    return model, model.visual.transform, tokenize
