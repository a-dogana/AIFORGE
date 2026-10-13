import os
import torch
import json
from typing import Dict, Any, Optional
from aiforge.core.config import ExperimentConfig

class LLMEngine:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm_config = config.architecture.llm_config
        
        if self.llm_config is None:
            raise ValueError("LLMConfig is required for LLMEngine.")
            
        self.device = config.device
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
        self.model = None
        self.tokenizer = None
        
    def load(self, local_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = local_path if local_path else self.llm_config.model_id
        
        # Security: Do not trust remote code by default
        trust_remote = False
        
        print(f"Loading LLM tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            local_files_only=not self.llm_config.allow_download,
            trust_remote_code=trust_remote
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Loading LLM model from {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=not self.llm_config.allow_download,
            trust_remote_code=trust_remote,
            device_map=self.device
        )
        
        if self.llm_config.lora_enabled and not local_path:
            # Only inject LoRA if it's a fresh base model load
            from peft import get_peft_model, LoraConfig, TaskType
            
            print(f"Injecting LoRA adapters (r={self.llm_config.lora_r}, alpha={self.llm_config.lora_alpha})...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=self.llm_config.lora_r, 
                lora_alpha=self.llm_config.lora_alpha, 
                lora_dropout=self.llm_config.lora_dropout
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")
            
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train(self, dataset_path: str):
        if not self.llm_config.lora_enabled:
            raise ValueError("Training requires LoRA to be enabled to protect base weights.")
            
        import math
        import time
        from datasets import load_dataset
        
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        
        # Determine train/val/test split
        # We need a proper split. Let's do 80/10/10.
        if len(dataset) < 3:
            raise ValueError("Dataset is too small to split into train/val/test.")
            
        # Optional: shuffle dataset
        dataset = dataset.shuffle(seed=self.config.seed)
        
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = max(1, int(0.1 * len(dataset)))
        
        train_ds = dataset.select(range(train_size))
        val_ds = dataset.select(range(train_size, min(len(dataset), train_size + val_size)))
        
        test_start = train_size + val_size
        if test_start < len(dataset):
            test_ds = dataset.select(range(test_start, len(dataset)))
        else:
            # Fallback for tiny datasets
            test_ds = dataset.select(range(0))
            
        print(f"Dataset splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
        
        def tokenize_and_mask(example):
            # Format: instruction + input -> output
            prompt = example.get("instruction", "") + "\n" + example.get("input", "")
            response = example.get("output", "")
            
            full_text = prompt + "\n" + response + self.tokenizer.eos_token
            
            # Tokenize prompt and full text
            prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.llm_config.max_seq_length)["input_ids"]
            full_tokens = self.tokenizer(full_text, truncation=True, max_length=self.llm_config.max_seq_length)
            
            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]
            
            # Mask the prompt part in labels
            labels = input_ids.copy()
            prompt_len = len(prompt_tokens)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
                
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            
        train_encoded = train_ds.map(tokenize_and_mask, remove_columns=train_ds.column_names)
        val_encoded = val_ds.map(tokenize_and_mask, remove_columns=val_ds.column_names)
        
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            input_ids = [torch.tensor(x["input_ids"]) for x in batch]
            attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
            labels = [torch.tensor(x["labels"]) for x in batch]
            
            # Pad
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            
        train_loader = DataLoader(train_encoded, batch_size=self.config.batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_encoded, batch_size=self.config.batch_size, collate_fn=collate_fn)
        
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        history = {"train_loss": [], "val_loss": []}
        
        # Mathematical Checks
        print("Running mathematical checks on parameters...")
        base_params_with_grad = 0
        lora_params_with_grad = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "lora" in name.lower():
                    lora_params_with_grad += 1
                else:
                    base_params_with_grad += 1
                    
        print(f"Trainable base params: {base_params_with_grad}")
        print(f"Trainable LoRA params: {lora_params_with_grad}")
        
        if base_params_with_grad > 0:
            raise RuntimeError("Base parameters should not require gradients when using LoRA.")
        if lora_params_with_grad == 0:
            raise RuntimeError("LoRA parameters are not receiving gradients.")
            
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            steps = 0
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                if not math.isfinite(loss.item()):
                    raise RuntimeError("Loss is NaN or Infinite. Training failed.")
                    
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
                
            avg_train_loss = epoch_loss / max(1, steps)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    val_steps += 1
                    
            avg_val_loss = val_loss / max(1, val_steps)
            history["val_loss"].append(avg_val_loss)
            print(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")
        
        # Test evaluation
        print("Evaluating on test set...")
        test_results = self.evaluate_test_set(test_ds)
        
        return {"history": history, "test_results": test_results, "splits": {"train": train_size, "val": val_size, "test": len(test_ds)}}
        
    def evaluate_test_set(self, test_ds):
        self.model.eval()
        results = []
        exact_matches = 0
        
        with torch.no_grad():
            for example in test_ds:
                prompt = example.get("instruction", "") + "\n" + example.get("input", "")
                expected = example.get("output", "")
                
                generated = self.generate(prompt, max_new_tokens=50)
                # Cleanup generated
                generated_clean = generated[len(prompt):].strip()
                expected_clean = expected.strip()
                
                match = (generated_clean == expected_clean)
                if match:
                    exact_matches += 1
                    
                results.append({
                    "prompt": prompt,
                    "expected": expected_clean,
                    "generated": generated_clean,
                    "match": match
                })
                
        metrics = {
            "exact_match_ratio": exact_matches / max(1, len(test_ds)),
            "total_evaluated": len(test_ds)
        }
        print(f"Test Set Evaluation: {metrics}")
        return metrics

    def save_adapters(self, save_dir: str):
        if not self.llm_config.lora_enabled:
            raise ValueError("Only LoRA adapters can be saved in this mode.")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Saved LoRA adapters to {save_dir}")
