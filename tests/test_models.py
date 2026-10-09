import pytest
import torch
from aiforge.core.config import ExperimentConfig
from aiforge.models import ArchitectureEngine
from aiforge.core.utils import set_seed

def test_model_reproducibility():
    set_seed(42)
    config = ExperimentConfig(
        data={"type": "digits_builtin"},
        architecture={
            "type": "mlp",
            "mlp_config": {
                "input_size": 10,
                "hidden_sizes": [8],
                "output_size": 2
            }
        }
    )
    
    model = ArchitectureEngine.get_model(config)
    model.eval()
    
    # Generate dummy input
    dummy_input = torch.randn(5, 10)
    
    # Get original predictions
    with torch.no_grad():
        original_outputs = model(dummy_input)
        
    # Save state dict
    state_dict = model.state_dict()
    
    # Recreate model
    set_seed(42)
    new_model = ArchitectureEngine.get_model(config)
    
    # Verify initial predictions differ (since weights are re-initialized randomly if no seed, but with seed they might be same. 
    # Actually with seed 42 they will be identical if re-initialized, so let's perturb the new model to be sure or just load the weights and check equality)
    # Let's just load the weights to be rigorous.
    new_model.load_state_dict(state_dict)
    new_model.eval()
    
    with torch.no_grad():
        loaded_outputs = new_model(dummy_input)
        
    # Compare outputs with a tolerance suitable for float32
    assert torch.allclose(original_outputs, loaded_outputs, atol=1e-6)
