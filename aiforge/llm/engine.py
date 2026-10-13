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
            
        is_qlora = getattr(self.llm_config, "qlora", False)
        
        # Check for QLoRA adapter in local_path
        qlora_weights_path = os.path.join(local_path, "qlora_adapter.pt") if local_path and os.path.exists(os.path.join(local_path, "qlora_adapter.pt")) else None

        if is_qlora and not local_path:
            # Check if bitsandbytes is available on CUDA
            has_bnb = False
            try:
                import bitsandbytes as bnb
                has_bnb = True
            except ImportError:
                has_bnb = False

            if has_bnb and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=getattr(self.llm_config, "quantization_type", "nf4"),
                    bnb_4bit_use_double_quant=getattr(self.llm_config, "double_quant", True),
                    bnb_4bit_compute_dtype=torch.float16
                )
                print(f"Loading LLM model with hardware 4-bit bitsandbytes...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    local_files_only=not self.llm_config.allow_download,
                    trust_remote_code=trust_remote,
                    device_map="auto"
                )
                self.model = prepare_model_for_kbit_training(self.model)
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.llm_config.lora_r,
                    lora_alpha=self.llm_config.lora_alpha,
                    lora_dropout=self.llm_config.lora_dropout
                )
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
            else:
                # Native AIFORGE NF4-QLoRA Engine (Optimized for Apple Silicon MPS & CPU)
                from aiforge.llm.qlora import apply_qlora_to_model
                print(f"Applying Native AIFORGE 4-bit NF4 QLoRA on {self.device}...")
                dtype = torch.float16 if self.device == "mps" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    local_files_only=not self.llm_config.allow_download,
                    trust_remote_code=trust_remote,
                    torch_dtype=dtype
                )
                stats = apply_qlora_to_model(
                    self.model,
                    lora_r=self.llm_config.lora_r,
                    lora_alpha=self.llm_config.lora_alpha,
                    lora_dropout=self.llm_config.lora_dropout,
                    quant_type=getattr(self.llm_config, "quantization_type", "nf4")
                )
                print(f"QLoRA Active: {stats['replaced_layers']} layers converted to 4-bit NF4. Trainable params: {stats['trainable_params']} ({stats['trainable_percentage']:.2f}%).")
        elif qlora_weights_path:
            # Loading saved native QLoRA adapter
            from aiforge.llm.qlora import apply_qlora_to_model, load_qlora_state_dict
            cfg_path = os.path.join(local_path, "qlora_config.json")
            cfg_data = {}
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg_data = json.load(f)
            base_id = cfg_data.get("model_id", self.llm_config.model_id)
            print(f"Loading base model {base_id} for QLoRA adapter...")
            dtype = torch.float16 if self.device == "mps" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                base_id,
                local_files_only=not self.llm_config.allow_download,
                trust_remote_code=trust_remote,
                torch_dtype=dtype
            )
            apply_qlora_to_model(
                self.model,
                lora_r=cfg_data.get("lora_r", self.llm_config.lora_r),
                lora_alpha=cfg_data.get("lora_alpha", self.llm_config.lora_alpha),
                lora_dropout=cfg_data.get("lora_dropout", self.llm_config.lora_dropout),
                quant_type=cfg_data.get("quant_type", "nf4")
            )
            adapter_weights = torch.load(qlora_weights_path, map_location="cpu", weights_only=True)
            loaded = load_qlora_state_dict(self.model, adapter_weights)
            print(f"Loaded {loaded} QLoRA adapter layers from {local_path}.")
        else:
            print(f"Loading LLM model from {model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=not self.llm_config.allow_download,
                trust_remote_code=trust_remote,
                device_map=self.device
            )
            
            if self.llm_config.lora_enabled and not local_path:
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
        if not (self.llm_config.lora_enabled or getattr(self.llm_config, "qlora", False)):
            raise ValueError("Training requires LoRA or QLoRA to be enabled to protect base weights.")
            
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
        is_qlora = getattr(self.llm_config, "qlora", False)
        if not (self.llm_config.lora_enabled or is_qlora):
            raise ValueError("Only LoRA or QLoRA adapters can be saved in this mode.")
        os.makedirs(save_dir, exist_ok=True)
        
        if is_qlora:
            from aiforge.llm.qlora import extract_qlora_state_dict
            qlora_state = extract_qlora_state_dict(self.model)
            if qlora_state:
                torch.save(qlora_state, os.path.join(save_dir, "qlora_adapter.pt"))
                cfg_path = os.path.join(save_dir, "qlora_config.json")
                with open(cfg_path, "w") as f:
                    json.dump({
                        "model_id": self.llm_config.model_id,
                        "lora_r": self.llm_config.lora_r,
                        "lora_alpha": self.llm_config.lora_alpha,
                        "lora_dropout": self.llm_config.lora_dropout,
                        "quant_type": getattr(self.llm_config, "quantization_type", "nf4"),
                        "qlora": True
                    }, f, indent=2)
                print(f"Saved native QLoRA adapters to {save_dir}")
        
        if hasattr(self.model, "save_pretrained"):
            try:
                self.model.save_pretrained(save_dir)
            except Exception:
                pass
                
        self.tokenizer.save_pretrained(save_dir)
        print(f"Saved tokenizer and adapters to {save_dir}")
