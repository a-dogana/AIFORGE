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


def test_chunked_quantization_equivalence():
    torch.manual_seed(7)
    w = torch.randn(97, 131)  # non-multiple of block_size
    p_full, s_full, sh1, pad1 = quantize_to_nf4(w, block_size=64, quant_type="nf4", chunk_blocks=4096)
    p_chunk, s_chunk, sh2, pad2 = quantize_to_nf4(w, block_size=64, quant_type="nf4", chunk_blocks=3)
    assert sh1 == sh2
    assert pad1 == pad2
    assert torch.equal(p_full, p_chunk)
    assert torch.allclose(s_full, s_chunk)


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
    assert not hasattr(q_layer, "_cached_dequant_w")
    
    x = torch.randn(4, in_features, requires_grad=True)
    out = q_layer(x)
    assert out.shape == (4, out_features)
    assert not hasattr(q_layer, "_cached_dequant_w")
    
    loss = out.sum()
    loss.backward()
    
    assert q_layer.lora_A.grad is not None
    assert q_layer.lora_B.grad is not None
    assert q_layer.lora_A.grad.shape == (in_features, 8)
    assert q_layer.lora_B.grad.shape == (8, out_features)
    assert q_layer.packed_weights.grad is None

    # Verify eval mode also never caches persistent full float weights
    q_layer.eval()
    _ = q_layer(x.detach())
    assert getattr(q_layer, "_cached_dequant_w", None) is None


def test_nf4_autograd_matches_reference_and_does_not_save_full_weight():
    """
    Verifies that _NF4BaseLinearFunction produces the exact same forward output
    and backward gradients (with respect to x, lora_A, and lora_B across stacked layers)
    as a reference implementation using F.linear on dequantized weights, while NOT
    retaining any full-sized (out_features, in_features) float tensor in saved_tensors.
    """
    torch.manual_seed(123)
    in_f, hidden_f, out_f = 64, 96, 48
    lin1 = nn.Linear(in_f, hidden_f, bias=True)
    lin2 = nn.Linear(hidden_f, out_f, bias=False)

    q1 = NF4QuantizedLinear.from_linear(lin1, lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
    q2 = NF4QuantizedLinear.from_linear(lin2, lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
    with torch.no_grad():
        q1.lora_B.normal_(0, 0.1)
        q2.lora_B.normal_(0, 0.1)

    # Build exact reference layers from dequantized base weights + identical LoRA params
    w1_deq = q1.dequantize()
    b1_deq = q1.bias_val.clone()
    lora_A1_ref = nn.Parameter(q1.lora_A.detach().clone())
    lora_B1_ref = nn.Parameter(q1.lora_B.detach().clone())

    w2_deq = q2.dequantize()
    lora_A2_ref = nn.Parameter(q2.lora_A.detach().clone())
    lora_B2_ref = nn.Parameter(q2.lora_B.detach().clone())

    x_q = torch.randn(3, 5, in_f, requires_grad=True)
    x_ref = x_q.detach().clone().requires_grad_(True)

    # Forward through stacked QLoRA layers
    h_q = q1(x_q)
    out_q = q2(h_q)

    # Inspect autograd saved tensors on the _NF4BaseLinearFunctionBackward nodes
    # Neither q1 nor q2 base backward nodes may store a tensor of shape (hidden_f, in_f) or (out_f, hidden_f)
    visited = set()
    stack = [out_q.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in visited:
            continue
        visited.add(fn)
        if "NF4BaseLinearFunctionBackward" in type(fn).__name__:
            saved = getattr(fn, "saved_tensors", ())
            for t in saved:
                assert t.shape not in {(hidden_f, in_f), (out_f, hidden_f)}
        for next_fn, _ in getattr(fn, "next_functions", ()):
            stack.append(next_fn)

    # Reference forward
    h_ref = torch.nn.functional.linear(x_ref, w1_deq, b1_deq) + ((x_ref @ lora_A1_ref) @ lora_B1_ref) * q1.scaling
    out_ref = torch.nn.functional.linear(h_ref, w2_deq, None) + ((h_ref @ lora_A2_ref) @ lora_B2_ref) * q2.scaling

    assert torch.allclose(out_q, out_ref, atol=1e-5)

    loss_q = (out_q ** 2).mean()
    loss_ref = (out_ref ** 2).mean()
    loss_q.backward()
    loss_ref.backward()

    assert torch.allclose(x_q.grad, x_ref.grad, atol=1e-5)
    assert torch.allclose(q1.lora_A.grad, lora_A1_ref.grad, atol=1e-5)
    assert torch.allclose(q1.lora_B.grad, lora_B1_ref.grad, atol=1e-5)
    assert torch.allclose(q2.lora_A.grad, lora_A2_ref.grad, atol=1e-5)
    assert torch.allclose(q2.lora_B.grad, lora_B2_ref.grad, atol=1e-5)


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
    
    # Verify actual memory footprint of buffers + parameters is much smaller than 3 * 64 * 64 * 4 bytes
    orig_linear_bytes = 3 * (64 * 64 * 4 + 64 * 4)
    qlora_bytes = sum(
        t.numel() * t.element_size()
        for m in (model.q_proj, model.v_proj, model.out_proj)
        for t in list(m.parameters()) + list(m.buffers())
    )
    assert qlora_bytes < orig_linear_bytes * 0.55

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


def test_qlora_import_and_type_annotations():
    import importlib
    import typing
    import aiforge.llm.qlora as qlora_mod

    reloaded = importlib.reload(qlora_mod)
    assert hasattr(reloaded, "Any")
    hints_conv1d = typing.get_type_hints(reloaded.NF4QuantizedLinear.from_conv1d)
    assert hints_conv1d["conv1d"] is typing.Any
    hints_apply = typing.get_type_hints(reloaded.apply_qlora_to_model)
    assert "return" in hints_apply
