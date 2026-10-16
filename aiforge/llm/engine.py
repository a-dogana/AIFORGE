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
            
        from aiforge.core.resources import select_device
        self.device = select_device(config.device)
                
        self.model = None
        self.tokenizer = None
        
    def load(self, local_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if local_path and os.path.isfile(local_path):
            local_path = os.path.dirname(local_path)
            
        model_id = local_path if local_path else self.llm_config.model_id
        
        # Security: Do not trust remote code by default
        trust_remote = False
        
        qlora_weights_path = (
            os.path.join(local_path, "qlora_adapter.pt")
            if local_path and os.path.exists(os.path.join(local_path, "qlora_adapter.pt"))
            else None
        )
        peft_cfg_path = (
            os.path.join(local_path, "adapter_config.json")
            if local_path and os.path.exists(os.path.join(local_path, "adapter_config.json"))
            else None
        )
        has_full_cfg = bool(local_path and os.path.exists(os.path.join(local_path, "config.json")))

        # Resolve tokenizer source (fallback to base model if adapter dir lacks tokenizer files)
        tok_source = model_id
        if local_path and not any(
            os.path.exists(os.path.join(local_path, fname))
            for fname in ("tokenizer_config.json", "tokenizer.json", "vocab.json")
        ):
            tok_source = self.llm_config.model_id

        print(f"Loading LLM tokenizer from {tok_source}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_source, 
            local_files_only=not self.llm_config.allow_download,
            trust_remote_code=trust_remote
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        is_qlora = getattr(self.llm_config, "qlora", False)

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
        elif peft_cfg_path and not has_full_cfg:
            # Loading saved PEFT LoRA adapter directory
            from peft import PeftModel
            with open(peft_cfg_path, "r") as f:
                peft_meta = json.load(f)
            base_id = peft_meta.get("base_model_name_or_path") or self.llm_config.model_id
            print(f"Loading base model {base_id} and PEFT adapter from {local_path}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                local_files_only=not self.llm_config.allow_download,
                trust_remote_code=trust_remote
            )
            self.model = PeftModel.from_pretrained(base_model, local_path)
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

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        messages: Optional[ list ] = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")
            
        self.model.eval()
        rendered_prompt = prompt
        if messages is not None:
            from aiforge.llm.sft_data import ensure_chat_template
            ensure_chat_template(self.tokenizer)
            rendered_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif getattr(self.tokenizer, "chat_template", None) and not any(
            marker in prompt for marker in ("<|im_start|>", "[INST]", "<|begin_of_text|>", "<|user|>")
        ):
            try:
                rendered_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered_prompt = prompt

        raw_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        inputs = (
            raw_inputs.to(self.device)
            if hasattr(raw_inputs, "to")
            else {k: v.to(self.device) for k, v in raw_inputs.items()}
        )
        input_len = inputs["input_ids"].shape[1]
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if temperature > 0.05:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                **gen_kwargs
            )
            
        generated_tokens = outputs[0][input_len:]
        out_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        # Fallback if empty
        if not out_text:
            out_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return out_text

    def _encode_sft_dataset(self, raw_rows):
        """Validate schema and encode all rows using the shared SFT chat-template encoder."""
        from aiforge.llm.sft_data import (
            CHATML_TEMPLATE,
            detect_dataset_schema,
            encode_example,
            ensure_chat_template,
        )

        schema = detect_dataset_schema(raw_rows)
        custom_tpl = (self.config.backend_args or {}).get("chat_template")
        if custom_tpl and custom_tpl not in ("auto", "chatml"):
            self.tokenizer.chat_template = custom_tpl
        elif custom_tpl == "chatml" and not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = CHATML_TEMPLATE
        else:
            ensure_chat_template(self.tokenizer)

        encoded_rows = []
        for idx, ex in enumerate(raw_rows):
            enc = encode_example(self.tokenizer, ex, self.llm_config.max_seq_length)
            encoded_rows.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": enc["labels"],
                    "_raw_index": idx,
                }
            )
        return schema, encoded_rows

    def train(self, dataset_path: str):
        if not (self.llm_config.lora_enabled or getattr(self.llm_config, "qlora", False)):
            raise ValueError("Training requires LoRA or QLoRA to be enabled to protect base weights.")
            
        import math
        import time
        from datasets import load_dataset
        
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        
        if len(dataset) < 3:
            raise ValueError("Dataset is too small to split into train/val/test.")
            
        dataset = dataset.shuffle(seed=self.config.seed)
        raw_list = [dataset[i] for i in range(len(dataset))]
        schema, encoded_all = self._encode_sft_dataset(raw_list)
        print(f"Detected SFT dataset schema: '{schema}' ({len(encoded_all)} examples)")

        train_size = max(1, int(0.8 * len(encoded_all)))
        val_size = max(1, int(0.1 * len(encoded_all)))
        
        train_encoded = encoded_all[:train_size]
        val_encoded = encoded_all[train_size : min(len(encoded_all), train_size + val_size)]
        
        test_start = train_size + val_size
        if test_start < len(encoded_all):
            test_indices = [x["_raw_index"] for x in encoded_all[test_start:]]
            test_ds = [raw_list[i] for i in test_indices]
        else:
            test_ds = []
            
        print(f"Dataset splits: Train={len(train_encoded)}, Val={len(val_encoded)}, Test={len(test_ds)}")
        
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
            attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
            labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
            
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
        
        return {
            "history": history,
            "test_results": test_results,
            "schema": schema,
            "splits": {"train": len(train_encoded), "val": len(val_encoded), "test": len(test_ds)},
        }
        
    def evaluate_test_set(self, test_ds):
        from aiforge.llm.sft_data import split_for_generation

        self.model.eval()
        results = []
        exact_matches = 0
        
        with torch.no_grad():
            for example in test_ds:
                prompt_msgs, expected = split_for_generation(example)
                generated = self.generate("", max_new_tokens=50, messages=prompt_msgs)
                generated_clean = generated.strip()
                expected_clean = expected.strip()
                
                match = (generated_clean == expected_clean)
                if match:
                    exact_matches += 1
                    
                results.append({
                    "prompt": prompt_msgs,
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
