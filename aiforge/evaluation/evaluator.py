import torch
import torch.nn as nn
from typing import Dict, Any
from sklearn.metrics import classification_report, confusion_matrix
from aiforge.core.config import ExperimentConfig
from aiforge.core.resources import ResourceChecker

class EvaluationEngine:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device_name = ResourceChecker.get_available_device(config.device)
        self.device = torch.device(self.device_name)
        
    def evaluate(self, model: nn.Module, test_loader) -> Dict[str, Any]:
        """
        Evaluates the model on the test set.
        """
        model = model.to(self.device)
        model.eval()
        
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
        if not all_targets:
            print("\n--- Evaluation Results ---")
            print("Warning: Test set is empty. Evaluation status: UNKNOWN")
            return {
                "accuracy": None,
                "classification_report": {},
                "confusion_matrix": [],
                "release_gate": "UNKNOWN",
            }

        # Calculate metrics
        report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_targets, all_predictions).tolist()
        
        import math
        raw_acc = report.get("accuracy")
        accuracy = float(raw_acc) if raw_acc is not None and math.isfinite(float(raw_acc)) else None
        print(f"\n--- Evaluation Results ---")
        if accuracy is not None:
            print(f"Test Accuracy: {accuracy:.4f}")
        else:
            print("Test Accuracy: UNKNOWN")
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }
