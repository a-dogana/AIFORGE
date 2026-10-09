import argparse
import sys
import os
import json
import torch
from aiforge.core.config import ExperimentConfig
from aiforge.core.resources import ResourceChecker
from aiforge.core.utils import set_seed
from aiforge.data.dataset import DataEngine
from aiforge.models import ArchitectureEngine
from aiforge.training.trainer import TrainingEngine
from aiforge.evaluation.evaluator import EvaluationEngine
from aiforge.experiments.registry import ExperimentRegistry

def run_experiment(args):
    print(f"=== AIFORGE: Running Experiment from {args.config} ===")
    try:
        config = ExperimentConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    set_seed(config.seed)
    ResourceChecker.print_resource_report()
    
    print("Loading data...")
    train_loader, val_loader, test_loader, artifacts = DataEngine.get_dataloaders(config)
    
    print("Initializing architecture...")
    model = ArchitectureEngine.get_model(config)
    student_params = sum(p.numel() for p in model.parameters())
    print(f"Student Model Parameters: {student_params}")
    
    teacher_engine = None
    if config.distillation.enabled:
        print(f"Initializing Teacher Engine (Experiment ID: {config.distillation.teacher_experiment_id})...")
        from aiforge.training.teacher import TeacherEngine
        teacher_engine = TeacherEngine(config.distillation.teacher_experiment_id, device=ResourceChecker.get_available_device(config.device))
        teacher_engine.validate_compatibility(config)
        teacher_params = teacher_engine.get_parameter_count()
        print(f"Teacher Model Parameters: {teacher_params}")
        if student_params >= teacher_params:
            print("Warning: Student model has more or equal parameters than the teacher model.")
    
    print("Starting training...")
    trainer = TrainingEngine(config)
    training_results = trainer.train(model, train_loader, val_loader, teacher_engine=teacher_engine)
    
    if teacher_engine:
        training_results["distillation_info"] = {
            "teacher_experiment_id": config.distillation.teacher_experiment_id,
            "teacher_parameters": teacher_params,
            "student_parameters": student_params,
            "temperature": config.distillation.temperature,
            "alpha": config.distillation.alpha
        }
    
    print("Evaluating model...")
    evaluator = EvaluationEngine(config)
    
    best_model_state = training_results.get("best_model_state")
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    evaluation_results = evaluator.evaluate(model, test_loader)
    
    print("Saving results and checkpoint to registry...")
    registry = ExperimentRegistry()
    registry.save_experiment(config, training_results, evaluation_results, device_used=trainer.device_name, preprocessing_artifacts=artifacts)
    print("=== Experiment Complete ===")

def list_experiments(args):
    registry = ExperimentRegistry()
    experiments = registry.list_experiments()
    if not experiments:
        print("No experiments found in the registry.")
        return
        
    print(f"{'ID':<10} | {'Date':<20} | {'Name':<30} | {'Test Acc':<10}")
    print("-" * 75)
    for exp in experiments:
        exp_id = exp.get("id", "N/A")
        date = exp.get("timestamp", "")[:19].replace("T", " ")
        name = exp.get("experiment_name", "N/A")
        
        # safely get accuracy
        try:
            acc = exp["results"]["evaluation"]["accuracy"]
            acc_str = f"{acc:.4f}"
        except KeyError:
            acc_str = "N/A"
            
        print(f"{exp_id:<10} | {date:<20} | {name:<30} | {acc_str:<10}")

def show_experiment(args):
    registry = ExperimentRegistry()
    try:
        record, _ = registry.load_experiment(args.id)
        # Pretty print JSON without the config to save space, or print all
        print(json.dumps(record, indent=2))
    except Exception as e:
        print(f"Error showing experiment: {e}")
        sys.exit(1)

def evaluate_experiment(args):
    print(f"=== AIFORGE: Evaluating Experiment {args.id} ===")
    registry = ExperimentRegistry()
    try:
        record, model_state = registry.load_experiment(args.id)
    except Exception as e:
        print(f"Error loading experiment: {e}")
        sys.exit(1)
        
    if model_state is None:
        print("Error: No model checkpoint found for this experiment.")
        sys.exit(1)
        
    config_dict = record.get("config", {})
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error parsing saved configuration: {e}")
        sys.exit(1)
        
    set_seed(config.seed)
    
    artifacts_saved = record.get("preprocessing_artifacts", {})
    
    # Reload data
    print("Loading test data...")
    _, _, test_loader, artifacts = DataEngine.get_dataloaders(config, saved_artifacts=artifacts_saved)
    
    # Rebuild model
    print("Rebuilding architecture...")
    model = ArchitectureEngine.get_model(config)
    model.load_state_dict(model_state)
    print("Model weights restored successfully.")
    
    # Evaluate
    print("Running evaluation...")
    evaluator = EvaluationEngine(config)
    results = evaluator.evaluate(model, test_loader)
    
    print("\nOriginal evaluation accuracy:", record["results"]["evaluation"].get("accuracy"))
    print("Current evaluation accuracy: ", results.get("accuracy"))

def resume_experiment(args):
    print(f"=== AIFORGE: Resuming Experiment {args.id} ===")
    registry = ExperimentRegistry()
    try:
        record, resume_state = registry.load_training_state(args.id)
    except Exception as e:
        print(f"Error loading training state: {e}")
        sys.exit(1)
        
    config_dict = record.get("config", {})
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error parsing saved configuration: {e}")
        sys.exit(1)
        
    set_seed(config.seed)
    
    artifacts_saved = record.get("preprocessing_artifacts", {})
    
    print("Loading data...")
    train_loader, val_loader, test_loader, artifacts = DataEngine.get_dataloaders(config, saved_artifacts=artifacts_saved)
    
    print("Rebuilding architecture...")
    model = ArchitectureEngine.get_model(config)
    
    print("Resuming training...")
    trainer = TrainingEngine(config)
    training_results = trainer.train(model, train_loader, val_loader, resume_state=resume_state)
    
    print("Evaluating model...")
    evaluator = EvaluationEngine(config)
    
    # Must evaluate the BEST model, so we load it explicitly from the results
    best_model_state = training_results.get("best_model_state")
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    evaluation_results = evaluator.evaluate(model, test_loader)
    
    print("Saving updated results and checkpoint to registry...")
    # Overwrite the same experiment ID
    registry.save_experiment(config, training_results, evaluation_results, device_used=trainer.device_name, preprocessing_artifacts=artifacts, experiment_id=args.id)
    print("=== Resume Complete ===")

def compare_experiments(args):
    print(f"=== AIFORGE: Comparing Experiments ===")
    registry = ExperimentRegistry()
    
    def measure_latency(model, sample_input):
        import time
        model.eval()
        with torch.no_grad():
            # warmup
            for _ in range(5): model(sample_input)
            start = time.time()
            for _ in range(100): model(sample_input)
            end = time.time()
        return (end - start) / 100.0 * 1000 # ms per batch
        
    for eid in [args.teacher_id, args.student_id]:
        record, model_state = registry.load_experiment(eid)
        config = ExperimentConfig(**record["config"])
        model = ArchitectureEngine.get_model(config)
        model.load_state_dict(model_state)
        params = sum(p.numel() for p in model.parameters())
        
        # sample input for latency
        _, _, test_loader, _ = DataEngine.get_dataloaders(config, saved_artifacts=record.get("preprocessing_artifacts", {}))
        sample_input, _ = next(iter(test_loader))
        latency = measure_latency(model, sample_input)
        
        print(f"\n--- Experiment: {eid} ---")
        print(f"Name: {record['experiment_name']}")
        print(f"Test Accuracy: {record['results']['evaluation'].get('accuracy', 'N/A')}")
        print(f"Parameters: {params}")
        print(f"Inference Latency (batch_size={config.batch_size}): {latency:.2f} ms")
        if "distillation_info" in record and record["distillation_info"]:
            print(f"Distilled from: {record['distillation_info'].get('teacher_experiment_id')}")

def run_tuning(args):
    print(f"=== AIFORGE: Running AutoTuning from {args.config} ===")
    try:
        base_config = ExperimentConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    if not base_config.tuning.enabled:
        print("Tuning is not enabled in this configuration.")
        sys.exit(1)
        
    set_seed(base_config.seed)
    ResourceChecker.print_resource_report()
    
    from aiforge.tuning.autotuner import AutoTuningEngine
    import optuna
    
    tuner = AutoTuningEngine(base_config)
    
    def objective_fn(config: ExperimentConfig) -> float:
        # Load data for this trial
        train_loader, val_loader, _, _ = DataEngine.get_dataloaders(config)
        
        # Architecture
        model = ArchitectureEngine.get_model(config)
        
        # Teacher
        teacher_engine = None
        if config.distillation.enabled:
            from aiforge.training.teacher import TeacherEngine
            teacher_engine = TeacherEngine(config.distillation.teacher_experiment_id, device=ResourceChecker.get_available_device(config.device))
            teacher_engine.validate_compatibility(config)
            
        # Training
        trainer = TrainingEngine(config)
        res = trainer.train(model, train_loader, val_loader, teacher_engine=teacher_engine)
        
        # Select objective metric
        metric = config.tuning.metric
        if metric == "val_loss":
            return min(res["history"]["val_loss"])
        elif metric == "val_accuracy":
            return max(res["history"]["val_accuracy"])
        else:
            # Try to get best metric from history
            if metric in res["history"]:
                if config.tuning.mode == "min":
                    return min(res["history"][metric])
                return max(res["history"][metric])
            raise ValueError(f"Unknown tuning metric {metric}")
            
    print("Starting optimization study...")
    study = tuner.run(objective_fn)
    
    print("\n=== Tuning Complete ===")
    print(f"Best trial value: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Evaluate best model
    if args.evaluate_best:
        print("\nTraining and evaluating the best configuration...")
        best_config = tuner.get_best_config()
        
        # Use run_experiment logic for the best configuration
        class DummyArgs:
            pass
        best_args = DummyArgs()
        best_args.config = args.config # Dummy, we'll override it
        
        # Monkey patch run_experiment config loading
        original_from_yaml = ExperimentConfig.from_yaml
        ExperimentConfig.from_yaml = lambda x: best_config
        try:
            run_experiment(best_args)
        finally:
            ExperimentConfig.from_yaml = original_from_yaml

def run_export(args):
    print(f"=== AIFORGE: Exporting Experiment {args.id} ===")
    from aiforge.deployment.deployer import DeploymentEngine
    
    deployer = DeploymentEngine()
    formats = args.formats.split(",")
    manifest = deployer.export(args.id, args.target, formats=formats)
    print("Deployment Manifest:")
    print(json.dumps(manifest, indent=2))

def run_deploy_compare(args):
    print(f"=== AIFORGE: Comparing Deployment Backends ===")
    from aiforge.deployment.inference import InferenceEngine
    import numpy as np
    import time
    
    # Check what backends are available
    with open(os.path.join(args.package, "manifest.json"), "r") as f:
        manifest = json.load(f)
        
    backends = manifest.get("formats", [])
    if "pytorch" not in backends:
        print("Error: PyTorch backend is required for baseline comparison.")
        sys.exit(1)
        
    # Create dummy input based on architecture
    arch = manifest.get("architecture")
    if arch == "mlp":
        # Load architecture to get input size
        with open(os.path.join(args.package, "architecture.json"), "r") as f:
            arch_dict = json.load(f)
        in_size = arch_dict["mlp_config"]["input_size"]
        dummy_input = np.random.randn(args.batch_size, in_size).astype(np.float32)
    elif arch == "cnn":
        with open(os.path.join(args.package, "architecture.json"), "r") as f:
            arch_dict = json.load(f)
        in_channels = arch_dict["cnn_config"]["in_channels"]
        # Use 32x32 as default for dummy
        dummy_input = torch.randn(args.batch_size, in_channels, 32, 32)
    else:
        print("Unsupported architecture for comparison.")
        sys.exit(1)
        
    results = {}
    
    for backend in backends:
        print(f"\n--- Backend: {backend.upper()} ---")
        try:
            engine = InferenceEngine(args.package, backend=backend)
            
            # Warmup
            for _ in range(5):
                engine.predict(dummy_input)
                
            # Benchmark
            start = time.time()
            for _ in range(100):
                out = engine.predict(dummy_input)
            end = time.time()
            
            latency = (end - start) / 100.0 * 1000
            print(f"Inference Latency: {latency:.2f} ms")
            
            results[backend] = out
            
        except Exception as e:
            print(f"Failed to run {backend}: {e}")
            
    # Compare numerical parity
    if len(results) > 1 and "pytorch" in results:
        pt_out = results["pytorch"]
        for backend, out in results.items():
            if backend == "pytorch": continue
            diff = np.abs(pt_out - out).max()
            print(f"\nMax difference between PyTorch and {backend.upper()}: {diff:.6e}")
            if diff < 1e-4:
                print(f"✅ {backend.upper()} matches PyTorch natively.")
            else:
                print(f"❌ {backend.upper()} diverges from PyTorch!")

def run_serve(args):
    print(f"=== AIFORGE: Starting API Server ===")
    from aiforge.serving.server import run_server
    run_server(args.package, host=args.host, port=args.port, backend=args.backend)

def run_llm_train(args):
    print(f"=== AIFORGE: Starting LLM Specialization ===")
    try:
        config = ExperimentConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    if config.architecture.type != "llm":
        print("Config architecture must be 'llm' for this command.")
        sys.exit(1)
        
    from aiforge.llm.engine import LLMEngine
    engine = LLMEngine(config)
    
    try:
        engine.load()
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        sys.exit(1)
        
    print("\n--- Baseline Inference ---")
    prompt = "def fibonacci(n):"
    print(f"Prompt: {prompt}")
    baseline_out = engine.generate(prompt, max_new_tokens=20)
    print(f"Output:\n{baseline_out}")
    
    print("\n--- Fine-Tuning ---")
    res = engine.train(args.dataset)
    print(f"Training completed. Final loss: {res['history']['train_loss'][-1]:.4f}")
    
    out_dir = "deploy_llm_adapter"
    engine.save_adapters(out_dir)
    
    print("\n--- Adapted Inference (Test Sample) ---")
    adapted_out = engine.generate(prompt, max_new_tokens=20)
    print(f"Output:\n{adapted_out}")
    
    print("\nLLM Specialization completed successfully.")

def run_serve_llm(args):
    print(f"=== AIFORGE: Starting LLM Generation Server ===")
    from aiforge.serving.llm_server import run_llm_server
    run_llm_server(args.config, adapter_dir=args.adapter, host=args.host, port=args.port)

def main():
    parser = argparse.ArgumentParser(description="AIFORGE CLI: Autonomous AI Laboratory")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # run command
    parser_run = subparsers.add_parser("run", help="Run an experiment from a YAML configuration")
    parser_run.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser_run.set_defaults(func=run_experiment)
    
    # tune command
    parser_tune = subparsers.add_parser("tune", help="Run hyperparameter tuning from a YAML configuration")
    parser_tune.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser_tune.add_argument("--evaluate-best", action="store_true", help="Train and evaluate the best configuration at the end")
    parser_tune.set_defaults(func=run_tuning)
    
    # list command
    parser_list = subparsers.add_parser("list", help="List all registered experiments")
    parser_list.set_defaults(func=list_experiments)
    
    # show command
    parser_show = subparsers.add_parser("show", help="Show details of a specific experiment")
    parser_show.add_argument("id", type=str, help="The experiment ID")
    parser_show.set_defaults(func=show_experiment)
    
    # evaluate command
    parser_eval = subparsers.add_parser("evaluate", help="Load and evaluate a saved model")
    parser_eval.add_argument("id", type=str, help="The experiment ID")
    parser_eval.set_defaults(func=evaluate_experiment)
    
    # resume command
    parser_resume = subparsers.add_parser("resume", help="Resume an interrupted experiment")
    parser_resume.add_argument("id", type=str, help="The experiment ID to resume")
    parser_resume.set_defaults(func=resume_experiment)
    
    # compare command
    parser_compare = subparsers.add_parser("compare", help="Compare a teacher and student experiment")
    parser_compare.add_argument("teacher_id", type=str, help="The teacher experiment ID")
    parser_compare.add_argument("student_id", type=str, help="The student experiment ID")
    parser_compare.set_defaults(func=compare_experiments)
    
    # export command
    parser_export = subparsers.add_parser("export", help="Export a trained model to a deployment package")
    parser_export.add_argument("id", type=str, help="The experiment ID")
    parser_export.add_argument("target", type=str, help="Target directory for the deployment package")
    parser_export.add_argument("--formats", type=str, default="pytorch", help="Comma-separated list of formats (e.g., pytorch,onnx,torchscript)")
    parser_export.set_defaults(func=run_export)
    
    # deploy-compare command
    parser_deploy = subparsers.add_parser("deploy-compare", help="Compare native vs exported inference backends")
    parser_deploy.add_argument("package", type=str, help="Path to the deployment package")
    parser_deploy.add_argument("--batch-size", type=int, default=32, help="Batch size for benchmark")
    parser_deploy.set_defaults(func=run_deploy_compare)
    
    # serve command
    parser_serve = subparsers.add_parser("serve", help="Start the HTTP Inference API Server")
    parser_serve.add_argument("package", type=str, help="Path to the deployment package")
    parser_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser_serve.add_argument("--backend", type=str, default="pytorch", help="Inference backend to use")
    parser_serve.set_defaults(func=run_serve)
    
    # llm-train command
    parser_llm = subparsers.add_parser("llm-train", help="Run an LLM Fine-Tuning experiment")
    parser_llm.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser_llm.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser_llm.set_defaults(func=run_llm_train)
    
    # serve-llm command
    parser_serve_llm = subparsers.add_parser("serve-llm", help="Start the HTTP LLM Generation Server")
    parser_serve_llm.add_argument("--config", type=str, required=True, help="Path to the base model YAML config")
    parser_serve_llm.add_argument("--adapter", type=str, help="Path to the adapted LoRA weights directory")
    parser_serve_llm.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to")
    parser_serve_llm.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser_serve_llm.set_defaults(func=run_serve_llm)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
