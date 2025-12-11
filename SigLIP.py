from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn.functional as F


class SigLIPModel(torch.nn.Module):
    def __init__(self, model_name="google/siglip2-large-patch16-512", use_compile: bool = False, embedding_dim: int = 1152):
        super().__init__()
        self.embedding_dim = embedding_dim
        # choose device and enable CUDA optimizations if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # load model with fp16 and SDPA attention for better performance
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.use_cuda else torch.float32,
            device_map="auto" if self.use_cuda else None,
            attn_implementation="sdpa"
        )
        if not self.use_cuda:
            self.model.to(self.device)
        self.model.eval()
        
        self.compiled = False
        if self.use_cuda and use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, backend="inductor")
                self.compiled = True
            except Exception:
                self.compiled = False

        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(self, image, texts: list):
        # SigLIP2 requires padding=max_length and max_length=64
        inputs = self.processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt")

        # move tensors to device; cast floating tensors to half when using CUDA
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and self.use_cuda:
                    inputs[k] = v.to(self.device, non_blocking=True).half()
                else:
                    inputs[k] = v.to(self.device, non_blocking=True)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.use_cuda):
                outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)  # probabilities

        output = []
        for i, text in enumerate(texts):
            # ensure scalar is on CPU before converting to Python float
            output.append((text, probs[0][i].cpu().item()))
        return output  # return outputs and confidence for input

    def encode_image(self, image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and self.use_cuda:
                    inputs[k] = v.to(self.device, non_blocking=True).half()
                else:
                    inputs[k] = v.to(self.device, non_blocking=True)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.use_cuda):
                outputs = self.model.vision_model(**inputs)
        embeds = outputs.pooler_output  # (1, hidden_dim)
        embeds = F.normalize(embeds, dim=-1)
        # Truncate or pad to match embedding_dim
        if embeds.shape[-1] >= self.embedding_dim:
            embeds = embeds[..., :self.embedding_dim]
        else:
            embeds = F.pad(embeds, (0, self.embedding_dim - embeds.shape[-1]))
        return embeds.squeeze(0).cpu().float()

    def encode_text(self, text: str) -> torch.Tensor:
        # SigLIP2 requires padding=max_length and max_length=64
        inputs = self.processor(text=[text], padding="max_length", max_length=64, return_tensors="pt")
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and self.use_cuda:
                    inputs[k] = v.to(self.device, non_blocking=True).half()
                else:
                    inputs[k] = v.to(self.device, non_blocking=True)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.use_cuda):
                outputs = self.model.text_model(**inputs)
        embeds = outputs.pooler_output  # (1, hidden_dim)
        embeds = F.normalize(embeds, dim=-1)
        # Truncate or pad to match embedding_dim
        if embeds.shape[-1] >= self.embedding_dim:
            embeds = embeds[..., :self.embedding_dim]
        else:
            embeds = F.pad(embeds, (0, self.embedding_dim - embeds.shape[-1]))
        return embeds.squeeze(0).cpu().float()
