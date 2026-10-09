import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from aiforge.core.config import ExperimentConfig
from aiforge.core.resources import ResourceChecker
import time
import copy

class TrainingEngine:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        requested = config.device
        self.device_name = ResourceChecker.get_available_device(requested)
        self.device = torch.device(self.device_name)
        print(f"TrainingEngine initialized. Using device: {self.device_name}")

    def train(self, model: nn.Module, train_loader, val_loader, resume_state: Optional[Dict[str, Any]] = None, teacher_engine=None) -> Dict[str, Any]:
        """
        Executes the training loop with Early Stopping, Best Checkpoint selection, and Scheduler.
        Optionally performs Knowledge Distillation if a TeacherEngine is provided.
        """
        import torch.nn.functional as F
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        scheduler = None
        if self.config.scheduler.enabled:
            if self.config.scheduler.type == "reduce_lr_on_plateau":
                mode = self.config.early_stopping.mode if self.config.early_stopping.metric == "val_loss" else "min"
                if self.config.early_stopping.enabled:
                   mode = self.config.early_stopping.mode
                   
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode=mode, 
                    factor=self.config.scheduler.factor,
                    patience=self.config.scheduler.patience,
                    min_lr=self.config.scheduler.min_lr
                )
        
        start_epoch = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": []
        }
        if teacher_engine:
            history["kd_loss"] = []
            history["supervised_loss"] = []
        
        best_metric_val = float('inf') if self.config.early_stopping.mode == 'min' else float('-inf')
        epochs_no_improve = 0
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stop_reason = "Completed all epochs."
        
        if resume_state:
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            start_epoch = resume_state["epoch"] + 1
            history = resume_state.get("history", history)
            best_metric_val = resume_state.get("best_metric_val", best_metric_val)
            epochs_no_improve = resume_state.get("epochs_no_improve", 0)
            best_epoch = resume_state.get("best_epoch", best_epoch)
            if "best_model_state" in resume_state:
                best_model_state = resume_state["best_model_state"]
            print(f"Resuming training from epoch {start_epoch}")

        start_time = time.time()
        
        for epoch in range(start_epoch, self.config.epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            running_kd = 0.0
            running_sup = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if teacher_engine:
                    teacher_logits = teacher_engine.get_logits(inputs)
                    T = self.config.distillation.temperature
                    alpha = self.config.distillation.alpha
                    
                    # KL Divergence: expected log(P) - expected log(Q) -> KLDivLoss expects input=log_prob, target=prob
                    kd_loss = nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(outputs / T, dim=1),
                        F.softmax(teacher_logits / T, dim=1)
                    ) * (T ** 2)
                    
                    supervised_loss = criterion(outputs, targets)
                    loss = alpha * kd_loss + (1 - alpha) * supervised_loss
                    running_kd += kd_loss.item() * inputs.size(0)
                    running_sup += supervised_loss.item() * inputs.size(0)
                else:
                    loss = criterion(outputs, targets)
                    
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(train_loader.dataset)
            history["train_loss"].append(epoch_loss)
            
            if teacher_engine:
                history["kd_loss"].append(running_kd / len(train_loader.dataset))
                history["supervised_loss"].append(running_sup / len(train_loader.dataset))
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = correct / len(val_loader.dataset)
            
            history["val_loss"].append(epoch_val_loss)
            history["val_accuracy"].append(epoch_val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            history["learning_rate"].append(current_lr)
            
            print(f"Epoch {epoch+1}/{self.config.epochs} - "
                  f"Train Loss: {epoch_loss:.4f} - "
                  f"Val Loss: {epoch_val_loss:.4f} - "
                  f"Val Acc: {epoch_val_acc:.4f} - LR: {current_lr:.6f}")
            
            # Checkpoint & Early Stopping logic
            metric_to_check = epoch_val_loss if self.config.early_stopping.metric == "val_loss" else epoch_val_acc
            mode = self.config.early_stopping.mode
            min_delta = self.config.early_stopping.min_delta
            
            is_better = False
            if mode == 'min':
                if metric_to_check < best_metric_val - min_delta:
                    is_better = True
            else: # mode == 'max'
                if metric_to_check > best_metric_val + min_delta:
                    is_better = True
                    
            if is_better:
                best_metric_val = metric_to_check
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if scheduler:
                scheduler.step(metric_to_check)
                
            # Early Stopping Check
            if self.config.early_stopping.enabled:
                if (epoch + 1) >= self.config.early_stopping.min_epochs:
                    if epochs_no_improve >= self.config.early_stopping.patience:
                        stop_reason = f"Early stopping triggered at epoch {epoch+1} (patience={self.config.early_stopping.patience})"
                        print(stop_reason)
                        break
                  
        end_time = time.time()
        
        # Prepare last state for resuming
        last_training_state = {
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "epoch": epoch if 'epoch' in locals() else start_epoch,
            "history": history,
            "best_metric_val": best_metric_val,
            "epochs_no_improve": epochs_no_improve,
            "best_epoch": best_epoch,
            "best_model_state": best_model_state
        }
        if scheduler:
            last_training_state["scheduler_state_dict"] = copy.deepcopy(scheduler.state_dict())
        
        return {
            "history": history,
            "training_time_seconds": end_time - start_time,
            "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else 0,
            "best_epoch": best_epoch,
            "best_metric": best_metric_val,
            "stop_reason": stop_reason,
            "total_epochs_run": epoch + 1 - start_epoch if 'epoch' in locals() else 0,
            "best_model_state": best_model_state,
            "last_training_state": last_training_state
        }
