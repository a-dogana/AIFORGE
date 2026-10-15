"""
AIFORGE QLoRA Engine (Quantized Low-Rank Adaptation).

Provides native 4-bit NormalFloat4 (NF4) and FP4 block-wise quantization with
trainable Low-Rank Adapters (LoRA) optimized for Apple Silicon (MPS), CPU, and CUDA.
Does not require external C/CUDA binary wheels for 4-bit execution.
"""

import math
import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union

# Official NF4 quantile levels from Dettmers et al. (2023) "QLoRA"
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955673218, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)

FP4_LEVELS = torch.tensor([
    -1.0, -0.75, -0.5, -0.25, -0.125, -0.0625, -0.03125, 0.0,
    0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0
], dtype=torch.float32)


def quantize_to_nf4(
    w: torch.Tensor,
    block_size: int = 64,
    quant_type: str = "nf4"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, int]:
    """
    Quantizes a floating-point weight tensor to packed 4-bit indices (2 values per uint8 byte)
    using block-wise NormalFloat4 or FP4 codebook.
    """
    codebook = NF4_LEVELS if quant_type == "nf4" else FP4_LEVELS
    codebook = codebook.to(w.device)

    shape = w.shape
    flat_w = w.detach().flatten().float()
    pad_len = (block_size - (flat_w.numel() % block_size)) % block_size
    if pad_len > 0:
        flat_w = torch.cat([flat_w, torch.zeros(pad_len, device=w.device, dtype=torch.float32)])

    blocks = flat_w.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    normed = blocks / scales

    # Distance to codebook levels
    dist = (normed.unsqueeze(-1) - codebook).abs()
    indices = dist.argmin(dim=-1).to(torch.uint8)

    # Bit-pack: 2 x 4-bit values into 1 x uint8
    even = indices[:, 0::2]
    odd = indices[:, 1::2]
    packed = (even << 4) | (odd & 0x0F)

    return packed, scales, shape, pad_len


def dequantize_from_nf4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: torch.Size,
    pad_len: int,
    quant_type: str = "nf4",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Dequantizes packed 4-bit indices back into a continuous floating-point tensor.
    """
    codebook = NF4_LEVELS if quant_type == "nf4" else FP4_LEVELS
    codebook = codebook.to(packed.device)

    even = (packed >> 4) & 0x0F
    odd = packed & 0x0F
    indices = torch.stack([even, odd], dim=2).view(scales.shape[0], -1)

    normed = codebook[indices.long()]
    dequant = (normed * scales).flatten()
    if pad_len > 0:
        dequant = dequant[:-pad_len]

    return dequant.view(shape).to(dtype)


class NF4QuantizedLinear(nn.Module):
    """
    A Linear layer whose base weight is frozen in 4-bit packed NF4 format,
    coupled with trainable low-rank adapters (LoRA).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        block_size: int = 64,
        quant_type: str = "nf4",
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r if lora_r > 0 else 1.0
        self.block_size = block_size
        self.quant_type = quant_type

        total_elements = in_features * out_features
        pad_len = (block_size - (total_elements % block_size)) % block_size
        num_blocks = (total_elements + pad_len) // block_size

        self.register_buffer("packed_weights", torch.zeros((num_blocks, block_size // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones((num_blocks, 1), dtype=torch.float32))
        self.pad_len = pad_len
        self.original_shape = (out_features, in_features)

        if bias:
            self.register_buffer("bias_val", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias_val", None)

        if lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty(in_features, lora_r))
            self.lora_B = nn.Parameter(torch.zeros(lora_r, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

        self._cached_dequant_w = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_dequant_w = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        block_size: int = 64,
        quant_type: str = "nf4"
    ) -> "NF4QuantizedLinear":
        out_f, in_f = linear.weight.shape
        has_bias = linear.bias is not None
        mod = cls(
            in_features=in_f,
            out_features=out_f,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            block_size=block_size,
            quant_type=quant_type,
            bias=has_bias
        )
        with torch.no_grad():
            w = linear.weight.data
            packed, scales, _, pad_len = quantize_to_nf4(w, block_size=block_size, quant_type=quant_type)
            mod.packed_weights.copy_(packed)
            mod.scales.copy_(scales)
            mod.pad_len = pad_len
            if has_bias:
                mod.bias_val.copy_(linear.bias.data.float())
        return mod

    @classmethod
    def from_conv1d(
        cls,
        conv1d: Any,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        block_size: int = 64,
        quant_type: str = "nf4"
    ) -> "NF4QuantizedLinear":
        # HuggingFace Conv1D has weight of shape (in_features, out_features)
        w = conv1d.weight.data.t()  # Transpose to (out_features, in_features)
        out_f, in_f = w.shape
        has_bias = getattr(conv1d, "bias", None) is not None
        mod = cls(
            in_features=in_f,
            out_features=out_f,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            block_size=block_size,
            quant_type=quant_type,
            bias=has_bias
        )
        with torch.no_grad():
            packed, scales, _, pad_len = quantize_to_nf4(w, block_size=block_size, quant_type=quant_type)
            mod.packed_weights.copy_(packed)
            mod.scales.copy_(scales)
            mod.pad_len = pad_len
            if has_bias:
                mod.bias_val.copy_(conv1d.bias.data.float())
        return mod

    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = dtype or torch.float32
        return dequantize_from_nf4(
            self.packed_weights,
            self.scales,
            self.original_shape,
            self.pad_len,
            quant_type=self.quant_type,
            dtype=target_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cached = getattr(self, "_cached_dequant_w", None)
        if not self.training:
            if (
                cached is None 
                or cached.dtype != x.dtype 
                or cached.device != x.device
            ):
                cached = self.dequantize(dtype=x.dtype).to(device=x.device, dtype=x.dtype)
                self._cached_dequant_w = cached
            w = cached
        else:
            w = self.dequantize(dtype=x.dtype).to(device=x.device, dtype=x.dtype)

        bias = self.bias_val.to(device=x.device, dtype=x.dtype) if self.bias_val is not None else None
        out = torch.nn.functional.linear(x, w, bias)

        if self.lora_r > 0 and self.lora_A is not None:
            # LoRA delta: (x * dropout) @ A @ B * scaling
            drop_x = self.dropout(x)
            lora_a = self.lora_A.to(device=x.device, dtype=x.dtype)
            lora_b = self.lora_B.to(device=x.device, dtype=x.dtype)
            lora_out = (drop_x @ lora_a) @ lora_b
            out = out + (lora_out * self.scaling)

        return out

    def to_standard_linear(self) -> nn.Linear:
        """
        Fuses the trained LoRA adapter back into the dequantized base weight,
        returning a standard standalone nn.Linear.
        """
        w_base = self.dequantize()
        if self.lora_r > 0 and self.lora_A is not None:
            # Add LoRA delta: delta_W = (A @ B).t()
            delta_w = (self.lora_A @ self.lora_B).t() * self.scaling
            w_base = w_base + delta_w

        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias_val is not None)
        with torch.no_grad():
            linear.weight.copy_(w_base)
            if self.bias_val is not None:
                linear.bias.copy_(self.bias_val)
        return linear


def apply_qlora_to_model(
    model: nn.Module,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    block_size: int = 64,
    quant_type: str = "nf4"
) -> Dict[str, Any]:
    """
    Recursively replaces target linear/conv1d layers in a model with NF4QuantizedLinear,
    freezing base weights and creating trainable LoRA adapters.
    """
    if target_modules is None:
        target_modules = [
            "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "out_proj",
            "fc1", "fc2", "dense", "linear", "gate_proj", "up_proj", "down_proj"
        ]

    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0
    total_base_params_compressed = 0

    def _replace_children(parent: nn.Module):
        nonlocal replaced_count, total_base_params_compressed
        for name, child in list(parent.named_children()):
            is_target = any(t in name.lower() for t in target_modules)
            is_linear = isinstance(child, nn.Linear)
            is_conv1d = child.__class__.__name__ == "Conv1D"

            if is_target and (is_linear or is_conv1d):
                if is_linear:
                    num_params = child.weight.numel()
                    new_mod = NF4QuantizedLinear.from_linear(
                        child,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        block_size=block_size,
                        quant_type=quant_type
                    )
                else:
                    num_params = child.weight.numel()
                    new_mod = NF4QuantizedLinear.from_conv1d(
                        child,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        block_size=block_size,
                        quant_type=quant_type
                    )

                setattr(parent, name, new_mod)
                replaced_count += 1
                total_base_params_compressed += num_params
            else:
                _replace_children(child)

    _replace_children(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_effective_params = total_base_params_compressed + sum(p.numel() for p in model.parameters())

    return {
        "replaced_layers": replaced_count,
        "base_params_compressed": total_base_params_compressed,
        "trainable_params": trainable_params,
        "total_params": total_effective_params,
        "trainable_percentage": (trainable_params / total_effective_params * 100.0) if total_effective_params > 0 else 0.0,
        "quant_type": quant_type
    }


def extract_qlora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts only the trained LoRA adapter weights from all NF4QuantizedLinear modules.
    Produces a lightweight adapter state dict (a few MBs instead of gigabytes).
    """
    adapter_state = {}
    for name, module in model.named_modules():
        if isinstance(module, NF4QuantizedLinear) and module.lora_A is not None:
            adapter_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            adapter_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
            adapter_state[f"{name}.scaling"] = torch.tensor(module.scaling)
    return adapter_state


def load_qlora_state_dict(model: nn.Module, adapter_state: Dict[str, torch.Tensor]) -> int:
    """
    Loads adapter weights into the NF4QuantizedLinear modules of the model.
    """
    loaded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, NF4QuantizedLinear) and module.lora_A is not None:
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in adapter_state and key_b in adapter_state:
                module.lora_A.data.copy_(adapter_state[key_a].to(module.lora_A.device))
                module.lora_B.data.copy_(adapter_state[key_b].to(module.lora_B.device))
                loaded_count += 1
    return loaded_count


def fuse_qlora_model(model: nn.Module) -> nn.Module:
    """
    Fuses all NF4QuantizedLinear modules back to standard nn.Linear layers with merged weights,
    ready for deployment or serving.
    """
    def _fuse_children(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, NF4QuantizedLinear):
                linear = child.to_standard_linear()
                setattr(parent, name, linear)
            else:
                _fuse_children(child)
    _fuse_children(model)
    return model
