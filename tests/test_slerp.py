import pytest
import torch
import math
from aiforge.core.merge_lab import slerp, dare_linear

def test_slerp_endpoints():
    # If alpha=0, slerp should return t0
    t0 = torch.tensor([1.0, 0.0, 0.0])
    t1 = torch.tensor([0.0, 1.0, 0.0])
    
    res_0 = slerp(t0, t1, alpha=0.0)
    assert torch.allclose(res_0, t0, atol=1e-5)
    
    # If alpha=1, slerp should return t1
    res_1 = slerp(t0, t1, alpha=1.0)
    assert torch.allclose(res_1, t1, atol=1e-5)

def test_slerp_orthogonal_midpoint():
    # Orthogonal vectors at 90 degrees: midpoint (alpha=0.5) should have norm 1 and equal components
    t0 = torch.tensor([1.0, 0.0])
    t1 = torch.tensor([0.0, 1.0])
    
    mid = slerp(t0, t1, alpha=0.5)
    expected = torch.tensor([math.cos(math.pi/4), math.sin(math.pi/4)])
    assert torch.allclose(mid, expected, atol=1e-5)
    assert abs(torch.norm(mid).item() - 1.0) < 1e-5

def test_slerp_collinear_fallback():
    # Parallel vectors should smoothly fallback to linear interpolation without NaN
    t0 = torch.tensor([2.0, 4.0])
    t1 = torch.tensor([2.0, 4.0])
    
    res = slerp(t0, t1, alpha=0.5)
    assert torch.allclose(res, t0, atol=1e-5)
    assert not torch.isnan(res).any()

def test_dare_linear_preserves_shape():
    t0 = torch.randn(5, 5)
    t1 = torch.randn(5, 5)
    
    res = dare_linear(t0, t1, alpha=0.5, drop_rate=0.2)
    assert res.shape == (5, 5)
    assert not torch.isnan(res).any()
