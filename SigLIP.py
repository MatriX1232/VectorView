from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn.functional as F


class SigLIPModel(torch.nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", use_compile: bool = False):
        super().__init__()
        # choose device and enable CUDA optimizations if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # load model and move to device; use fp16 on GPU for better throughput
        self.model = AutoModel.from_pretrained(model_name)
        if self.use_cuda:
            self.model.half().to(self.device, memory_format=torch.channels_last)
        else:
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
        inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt")

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
