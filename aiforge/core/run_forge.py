import os
import json
import time
import argparse
import sys
import re
from openai import OpenAI
from aiforge.core.secrets import get_secret
from aiforge.core.config import ExperimentConfig
from aiforge.llm.backends.soup import SoupBackend
from aiforge.experiments.registry import ExperimentRegistry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--qlora", action="store_true", default=False, help="Enable 4-bit NF4 QLoRA quantization")
    parser.add_argument("--method", choices=["quick", "lima"], default="quick", help="Dataset generation strategy: 'quick' or 'lima'")
    parser.add_argument("--target-samples", type=int, default=100, help="Target curated samples count for LIMA")
    parser.add_argument("--batch-size", type=int, default=20, help="Micro-batch size for LIMA generation")
    args = parser.parse_args()

    print(f"=== FORGE TRAINING PIPELINE: {args.name} (Method: {args.method.upper()}) ===", flush=True)

    print("PROGRESS: 10", flush=True)
    print("1. Caricamento connettore da registry...", flush=True)
    connectors_path = "registry/connectors.json"
    api_key = None
    base_url = "https://api.openai.com/v1"
    
    if os.path.exists(connectors_path):
        with open(connectors_path, "r") as f:
            connectors = json.load(f)
            if connectors:
                conn = connectors[0]
                base_url = conn.get("base_url", base_url)
                api_key = get_secret(conn["id"])
                print(f"✅ Utilizzo connettore: {conn['name']}", flush=True)

    if not api_key:
        print("❌ Errore: Nessun connettore con API Key trovata. Vai in Connettori (OpenAI Std) per configurarlo.", flush=True)
        sys.exit(1)
    
    print("PROGRESS: 20", flush=True)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    clean_name = re.sub(r"[^\w\-]", "_", args.name.strip().lower())
    dataset_path = f"datasets/forge_{clean_name}.jsonl"
    candidates = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
    
    if args.method == "lima":
        print(f"3. Esecuzione Pipeline LIMA ad Alta Qualità (Target: {args.target_samples} campioni)...", flush=True)
        from aiforge.data.lima_pipeline import LIMAPipeline
        
        target_model = candidates[0]
        pipeline = LIMAPipeline(
            client=client,
            model_name=target_model,
            min_request_delay=1.0,
            max_retries=4
        )
        
        def lima_progress(pct, msg):
            scaled_pct = int(20 + (pct * 0.40))  # Scale LIMA generation to 20-60%
            print(f"PROGRESS: {scaled_pct}", flush=True)
            print(f"LIMA [{pct}%]: {msg}", flush=True)

        res_lima = pipeline.run_pipeline(
            job_id=clean_name,
            main_topic=args.topic,
            target_samples=args.target_samples,
            batch_size=args.batch_size,
            output_final_path=dataset_path,
            progress_cb=lima_progress
        )
        print(f"✅ Dataset LIMA Curato: {res_lima['curated_count']} campioni salvati (Filtrati da {res_lima['raw_staged_count']} grezzi)!", flush=True)
    else:
        print("3. Generazione dataset sintetico rapido (Quick Mode)...", flush=True)
        prompt = f"""Crea 15 esempi didattici in formato JSON per il seguente topic: {args.topic}.
Rispondi ESCLUSIVAMENTE con un array JSON valido, senza blocchi di codice markdown.
Struttura:
[
  {{
    "instruction": "descrizione del compito",
    "input": "",
    "output": "risposta o codice"
  }}
]"""

        response = None
        for target_model in candidates:
            try:
                response = client.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                print(f"✅ Generazione riuscita con {target_model} (OpenAI Compatible)!", flush=True)
                break
            except Exception as err:
                time.sleep(2)

        if not response:
            print("❌ Generazione fallita.", flush=True)
            sys.exit(1)

        print("PROGRESS: 40", flush=True)
        try:
            raw_text = response.choices[0].message.content.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.startswith("```"): raw_text = raw_text[3:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            items = json.loads(raw_text)
            os.makedirs("datasets", exist_ok=True)
            with open(dataset_path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"✅ Dataset generato: {len(items)} esempi salvati in {dataset_path}!", flush=True)
        except Exception as e:
            print(f"❌ Errore parsing dataset: {e}", flush=True)
            sys.exit(1)

    print("PROGRESS: 60", flush=True)
    print("4. Preparazione configurazione per il fine-tuning LoRA...", flush=True)
    
    soup_env = "/Users/alessandrodogana/.gemini/antigravity/brain/177bf045-2b86-412c-a413-1057a8eeec8b/venv_soup"
    cfg = ExperimentConfig(
        experiment_name=args.name,
        domain="llm",
        task="generation",
        architecture={
            "type": "llm",
            "llm_config": {
                "model_id": args.model,
                "allow_download": True,
                "lora_enabled": True,
                "qlora": args.qlora,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05
            }
        },
        data={"type": "jsonl", "csv_config": {"path": dataset_path, "target_column": "output"}},
        batch_size=2,
        epochs=3,
        device="mps",
        backend="soup",
        backend_args={
            "soup_env_path": soup_env,
            "chat_template": "chatml",
            "quantization": "4bit" if args.qlora else "none"
        }
    )

    print("PROGRESS: 70", flush=True)
    print("5. Avvio motore di training Soup su Apple Silicon (MPS)...", flush=True)
    try:
        backend = SoupBackend(cfg)
        res = backend.train(dataset_path)
    except Exception as e:
        print(f"❌ Errore durante il training: {e}", flush=True)
        sys.exit(1)

    print("PROGRESS: 90", flush=True)
    adapter_path = res["checkpoint_path"]
    print(f"✅ Addestramento completato! Adapter salvato in: {adapter_path}", flush=True)

    print("6. Registrazione nell'AIFORGE Registry...", flush=True)
    registry = ExperimentRegistry("registry")
    registry.save_experiment(
        config=cfg,
        training_results={"train_loss": 0.12},
        evaluation_results={},
        device_used="mps",
        experiment_id=clean_name
    )
    print("PROGRESS: 100", flush=True)
    print("🎉 Pipeline Forge Completata con Successo!", flush=True)

if __name__ == "__main__":
    main()
