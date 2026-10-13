import pytest
from aiforge.core.config import ExperimentConfig
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.core.unified_training import UnifiedOrchestrator, ClassicalAdapter, LLMAdapter

def test_unified_orchestrator_routing_classical(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    config = ExperimentConfig(
        experiment_name="Test Classical",
        domain="tabular",
        task="classification",
        data={"type": "digits_builtin"},
        architecture={"type": "mlp", "mlp_config": {"input_size": 10, "hidden_sizes": [5], "output_size": 2}}
    )
    orchestrator = UnifiedOrchestrator(config, registry)
    assert isinstance(orchestrator.adapter, ClassicalAdapter)

def test_unified_orchestrator_routing_llm(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    config = ExperimentConfig(
        experiment_name="Test LLM",
        domain="llm",
        task="generation",
        data={"type": "csv", "csv_config": {"path": "dummy.csv", "target_column": "text"}},
        architecture={"type": "llm", "llm_config": {"model_id": "gpt2"}}
    )
    orchestrator = UnifiedOrchestrator(config, registry)
    assert isinstance(orchestrator.adapter, LLMAdapter)

def test_unified_orchestrator_routing_unsupported(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    config = ExperimentConfig(
        experiment_name="Test Invalid",
        domain="tabular",
        task="classification",
        data={"type": "digits_builtin"},
        architecture={"type": "quantum_nn"}
    )
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        UnifiedOrchestrator(config, registry)
