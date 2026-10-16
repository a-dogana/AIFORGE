import json
import os
import stat
import struct
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from aiforge.core.config import ExperimentConfig
from aiforge.core.unified_training import UnifiedOrchestrator
from aiforge.deployment.deployer import DeploymentEngine
from aiforge.deployment.gguf_export import GGUFExporter, validate_gguf_file
from aiforge.deployment.inference import InferenceEngine
from aiforge.evaluation.merge_eval import MergeEvaluator
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.llm.engine import LLMEngine
from aiforge.core.merge_lab import MergeLab, MergeRequest


def test_e2e_classical_train_registry_merge_eval_export_inference(tmp_path):
    """
    End-to-end integration test for Flow 1:
    training -> registry -> merge -> evaluation -> export -> inference
    """
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    # Create synthetic tabular CSV dataset with missing values
    rng = np.random.default_rng(42)
    n_samples = 80
    x1 = rng.normal(2.0, 1.0, size=n_samples).astype(np.float32)
    x2 = rng.normal(-1.0, 0.5, size=n_samples).astype(np.float32)
    x3 = rng.normal(0.5, 2.0, size=n_samples).astype(np.float32)
    y = ((x1 + x2 - 0.5 * x3) > 0.5).astype(int)
    x1[0] = np.nan
    x2[3] = np.nan

    csv_path = tmp_path / "tabular.csv"
    pd.DataFrame({"f1": x1, "f2": x2, "f3": x3, "target": y}).to_csv(csv_path, index=False)

    def _make_cfg(name: str, seed: int) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_name=name,
            seed=seed,
            device="cpu",
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            data={
                "type": "csv",
                "csv_config": {
                    "path": str(csv_path),
                    "target_column": "target",
                    "missing_values": "mean",
                },
            },
            architecture={
                "type": "mlp",
                "mlp_config": {"input_size": 3, "hidden_sizes": [8], "output_size": 2},
            },
        )

    # 1. Train Model A and Model B via UnifiedOrchestrator
    orch_a = UnifiedOrchestrator(_make_cfg("Classical A", seed=11), registry=registry)
    exp_a_id = orch_a.run()

    orch_b = UnifiedOrchestrator(_make_cfg("Classical B", seed=22), registry=registry)
    exp_b_id = orch_b.run()

    rec_a = registry.get_experiment_record(exp_a_id)
    assert rec_a is not None
    assert rec_a["checkpoint_path"].endswith(".pt")
    assert os.path.isfile(rec_a["checkpoint_path"])
    assert "imputation" in rec_a["preprocessing_artifacts"]
    assert "scaler" in rec_a["preprocessing_artifacts"]

    # 2. Merge Model A and Model B via MergeLab
    merge_lab = MergeLab(registry)
    merge_req = MergeRequest(
        strategy="linear_weight",
        model_a_id=exp_a_id,
        model_b_id=exp_b_id,
        alpha=0.4,
        output_name="Merged Classical AB",
    )
    preflight = merge_lab.validate_pre_merge(merge_req)
    assert preflight["status"] == "COMPATIBLE"

    merged_id = merge_lab.execute_merge(merge_req)
    rec_merged = registry.get_experiment_record(merged_id)
    assert rec_merged is not None
    assert rec_merged["checkpoint_path"].endswith(".pt")
    assert os.path.isfile(rec_merged["checkpoint_path"])

    # 3. Evaluate Merged Model via MergeEvaluator
    evaluator = MergeEvaluator(registry)
    eval_report = evaluator.evaluate_merge(exp_a_id, exp_b_id, merged_id)
    assert eval_report["architecture"] == "mlp"
    assert eval_report["release_gate"] in ("PROMOTED", "PASS", "FLAGGED_REGRESSION", "CRITICAL_REGRESSION")
    assert eval_report["release_gate"] != "UNKNOWN"

    # 4. Export Merged Model via DeploymentEngine
    deploy_dir = tmp_path / "deployed_merged"
    deployer = DeploymentEngine(registry_dir=str(reg_dir))
    manifest = deployer.export(merged_id, str(deploy_dir), formats=["pytorch", "torchscript"])
    assert "pytorch" in manifest["formats"]
    assert "imputation" in manifest["preprocessing_artifacts"]

    # 5. Run Inference via InferenceEngine (including NaN input handled by saved imputation)
    inf_engine = InferenceEngine(str(deploy_dir), backend="pytorch")
    sample_with_nan = np.array([[np.nan, -1.0, 0.5], [2.5, -0.5, np.nan]], dtype=np.float32)
    preds = inf_engine.predict(sample_with_nan)
    assert preds.shape == (2, 2)
    assert np.isfinite(preds).all()


def test_e2e_llm_conversational_qlora_and_gguf_export(tmp_path, monkeypatch):
    """
    End-to-end integration test for Flow 2 & Flow 3:
    Conversational SFT (`messages`) -> chat template -> filter truncated prompt-only samples ->
    1-step QLoRA training -> save adapter to registry -> reload & generate -> GGUF export & header validation.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    # Build a tiny local HuggingFace GPT-2 model + chat-templated tokenizer
    base_model_dir = tmp_path / "tiny_gpt2_base"
    base_model_dir.mkdir()

    vocab_words = [
        "<pad>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>",
        "system", "user", "assistant", "hello", "world", "how", "are",
        "you", "fine", "thanks", "long", "text", "token", "python", "code",
    ]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<eos>",
    )
    fast_tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|> {{ message['role'] }} {{ message['content'] }} <|im_end|> "
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
    )
    fast_tok.save_pretrained(str(base_model_dir))

    gpt_cfg = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_ctx=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        pad_token_id=vocab["<pad>"],
        eos_token_id=vocab["<eos>"],
        bos_token_id=vocab["<eos>"],
    )
    tiny_model = GPT2LMHeadModel(gpt_cfg)
    tiny_model.save_pretrained(str(base_model_dir))

    # Create a conversational JSONL dataset with 6 valid multi-turn examples and 1 truncated example
    jsonl_path = tmp_path / "conv_sft.jsonl"
    valid_conv = {
        "messages": [
            {"role": "system", "content": "you are assistant"},
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "fine thanks"},
        ]
    }
    long_prompt_conv = {
        "messages": [
            {"role": "user", "content": " ".join(["long"] * 60)},
            {"role": "assistant", "content": "fine thanks"},
        ]
    }
    rows = [valid_conv] * 6 + [long_prompt_conv]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    exp_cfg = ExperimentConfig(
        experiment_name="E2E Conversational QLoRA",
        domain="llm",
        task="generation",
        seed=42,
        device="cpu",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        data={
            "type": "jsonl",
            "csv_config": {"path": str(jsonl_path), "target_column": "messages"},
        },
        architecture={
            "type": "llm",
            "llm_config": {
                "model_id": str(base_model_dir),
                "max_seq_length": 32,
                "lora_enabled": True,
                "qlora": True,
                "quantization_bits": 4,
                "quantization_type": "nf4",
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "allow_download": False,
            },
        },
    )

    engine = LLMEngine(exp_cfg)
    engine.load()
    train_res = engine.train(str(jsonl_path))

    # The 1 overly-long prompt sample must have been filtered out before splitting
    stats = train_res["dataset_stats"]
    assert stats["initial"] == 7
    assert stats["discarded"] == 1
    assert stats["valid"] == 6
    splits = train_res["splits"]
    assert splits["train"] + splits["val"] + splits["test"] == 6

    adapter_save_dir = tmp_path / "saved_qlora_adapter"
    engine.save_adapters(str(adapter_save_dir))
    assert os.path.isfile(adapter_save_dir / "qlora_adapter.pt")
    assert os.path.isfile(adapter_save_dir / "qlora_config.json")

    exp_id = registry.save_experiment(
        exp_cfg,
        {"history": train_res["history"], "dataset_stats": stats},
        train_res["test_results"],
        "cpu",
        checkpoint_path=str(adapter_save_dir),
    )

    # Reload from registry via centralized loader and via LLMEngine.load, then generate with chat template
    loaded_model, loaded_tok = registry.load_llm_for_inference(exp_id, device="cpu")
    assert loaded_model is not None and loaded_tok is not None

    reloaded_engine = LLMEngine(exp_cfg)
    reloaded_engine.load(str(adapter_save_dir))
    reply = reloaded_engine.generate(
        prompt="",
        messages=[{"role": "user", "content": "hello world"}],
        max_new_tokens=4,
        temperature=0.0,
    )
    assert isinstance(reply, str)

    # Flow 3: GGUF export with adapter fusion + binary GGUF header validation
    convert_script = tmp_path / "convert_hf_to_gguf.py"
    convert_script.write_text(
        "import os, sys, struct\n"
        "args = sys.argv[1:]\n"
        "hf_dir = args[0]\n"
        "assert os.path.isfile(os.path.join(hf_dir, 'config.json')), 'Expected fused full HF model config.json'\n"
        "out_idx = args.index('--outfile') + 1\n"
        "outfile = args[out_idx]\n"
        "with open(outfile, 'wb') as f:\n"
        "    f.write(struct.pack('<4sIQQ', b'GGUF', 3, 12, 6) + b'GGUFBINARY')\n",
        encoding="utf-8",
    )

    exporter = GGUFExporter(
        registry=registry,
        convert_script_path=str(convert_script),
    )
    gguf_res = exporter.export(
        experiment_id=exp_id,
        quantization="F16",
        output_dir=str(tmp_path / "gguf_out"),
        run_conversion=True,
    )
    assert gguf_res["conversion_completed"] is True
    header_info = validate_gguf_file(gguf_res["target_gguf_path"])
    assert header_info["valid"] is True
    assert header_info["version"] == 3
    assert header_info["tensor_count"] == 12
