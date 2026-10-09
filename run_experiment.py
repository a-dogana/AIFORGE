import sys
from aiforge.core.config import ExperimentConfig
from aiforge.core.resources import ResourceChecker
from aiforge.data.dataset import DataEngine
from aiforge.models.mlp import ArchitectureEngine
from aiforge.training.trainer import TrainingEngine
from aiforge.evaluation.evaluator import EvaluationEngine
from aiforge.experiments.registry import ExperimentRegistry
import yaml

def main():
    print("=== AIFORGE: Running Experiment ===")
    
    # 1. Configuration Check
    # In a real app we load from YAML, here we use default for digits
    try:
        config = ExperimentConfig(
            dataset_name="digits",
            epochs=10,
            batch_size=32,
            learning_rate=0.005,
            hidden_sizes=[128, 64],
            input_size=64, # 8x8 images
            output_size=10
        )
    except Exception as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
        
    print(f"Configuration Validated: {config.domain} / {config.task}")
    
    # 2. Resource Check
    ResourceChecker.print_resource_report()
    
    # 3. Data Engine
    print("Loading data...")
    train_loader, val_loader, test_loader = DataEngine.get_dataloaders(config)
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    
    # 4. Architecture Engine
    print("Initializing architecture...")
    model = ArchitectureEngine.get_model(config)
    
    # 5. Training Engine
    print("Starting training...")
    trainer = TrainingEngine(config)
    training_results = trainer.train(model, train_loader, val_loader)
    
    # 6. Evaluation Engine
    print("Evaluating model...")
    evaluator = EvaluationEngine(config)
    evaluation_results = evaluator.evaluate(model, test_loader)
    
    # 7. Experiment Registry
    print("Saving results to registry...")
    registry = ExperimentRegistry()
    registry.save_experiment(config, training_results, evaluation_results)
    
    print("=== Experiment Complete ===")

if __name__ == "__main__":
    main()
