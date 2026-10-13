import torch
import torch.nn as nn
import pytest
from aiforge.llm.qlora import (
    quantize_to_nf4,
    dequantize_from_nf4,
    NF4QuantizedLinear,
    apply_qlora_to_model,
    extract_qlora_state_dict,
    load_qlora_state_dict,
    fuse_qlora_model
)

def test_nf4_quantization_numerical_stability():
    # Test random weight tensor
    torch.manual_seed(42)
    w = torch.randn(128, 256)
    orig_bytes = w.numel() * 4
    
    # NF4
    packed, scales, shape, pad_len = quantize_to_nf4(w, block_size=64, quant_type="nf4")
    packed_bytes = packed.numel() + scales.numel() * 4
    recon = dequantize_from_nf4(packed, scales, shape, pad_len, quant_type="nf4")
    
    compression_ratio = orig_bytes / packed_bytes
    assert compression_ratio > 6.5, f"Expected >6.5x compression, got {compression_ratio:.2f}x"
    assert recon.shape == w.shape
    mean_abs_err = (w - recon).abs().mean().item()
    assert mean_abs_err < 0.15, f"Reconstruction error too high: {mean_abs_err}"

    # FP4
    packed_fp4, scales_fp4, _, pad_len_fp4 = quantize_to_nf4(w, block_size=64, quant_type="fp4")
    recon_fp4 = dequantize_from_nf4(packed_fp4, scales_fp4, shape, pad_len_fp4, quant_type="fp4")
    assert recon_fp4.shape == w.shape


def test_nf4_quantized_linear_forward_backward():
    torch.manual_seed(42)
    in_features = 64
    out_features = 128
    base_linear = nn.Linear(in_features, out_features, bias=True)
    
    q_layer = NF4QuantizedLinear.from_linear(
        base_linear,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        block_size=64,
        quant_type="nf4"
    )
    
    assert q_layer.packed_weights.requires_grad is False
    assert q_layer.lora_A.requires_grad is True
    assert q_layer.lora_B.requires_grad is True
    
    x = torch.randn(4, in_features, requires_grad=True)
    out = q_layer(x)
    assert out.shape == (4, out_features)
    
    loss = out.sum()
    loss.backward()
    
    assert q_layer.lora_A.grad is not None
    assert q_layer.lora_B.grad is not None
    assert q_layer.lora_A.grad.shape == (in_features, 8)
    assert q_layer.lora_B.grad.shape == (8, out_features)
    assert q_layer.packed_weights.grad is None


def test_apply_qlora_to_model():
    class ToyTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.norm = nn.LayerNorm(64)
            
        def forward(self, x):
            return self.out_proj(self.v_proj(self.q_proj(self.norm(x))))
            
    model = ToyTransformerBlock()
    stats = apply_qlora_to_model(model, lora_r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj", "out_proj"])
    
    assert stats["replaced_layers"] == 3
    assert stats["trainable_params"] > 0
    assert stats["trainable_percentage"] < 50.0
    assert isinstance(model.q_proj, NF4QuantizedLinear)
    assert isinstance(model.v_proj, NF4QuantizedLinear)
    assert isinstance(model.out_proj, NF4QuantizedLinear)
    
    # Test forward pass
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 64)


def test_qlora_adapter_save_and_load():
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    base_linear = nn.Linear(in_features, out_features)
    
    model1 = NF4QuantizedLinear.from_linear(base_linear, lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
    # Modify adapter weights
    with torch.no_grad():
        model1.lora_A.add_(0.5)
        model1.lora_B.add_(0.2)
        
    wrapper1 = nn.Sequential(model1)
    adapter_state = extract_qlora_state_dict(wrapper1)
    assert len(adapter_state) > 0
    
    # Create fresh model with same base
    model2 = NF4QuantizedLinear.from_linear(base_linear, lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
    wrapper2 = nn.Sequential(model2)
    loaded_count = load_qlora_state_dict(wrapper2, adapter_state)
    assert loaded_count == 1
    
    # Verify outputs match
    x = torch.randn(2, in_features)
    out1 = wrapper1(x)
    out2 = wrapper2(x)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_qlora_fuse_to_standard_linear():
    torch.manual_seed(42)
    in_features, out_features = 32, 64
    base_linear = nn.Linear(in_features, out_features)
    
    q_layer = NF4QuantizedLinear.from_linear(base_linear, lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
    with torch.no_grad():
        q_layer.lora_A.fill_(0.1)
        q_layer.lora_B.fill_(0.1)
        
    x = torch.randn(2, in_features)
    q_out = q_layer(x)
    
    fused_linear = q_layer.to_standard_linear()
    assert isinstance(fused_linear, nn.Linear)
    fused_out = fused_linear(x)
    
    assert torch.allclose(q_out, fused_out, atol=1e-4)
