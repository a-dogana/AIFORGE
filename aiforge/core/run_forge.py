import os
import json
import time
import argparse
import sys
import re
from typing import Optional
from openai import OpenAI
from aiforge.core.secrets import get_secret
from aiforge.core.config import ExperimentConfig
from aiforge.llm.backends.soup import SoupBackend, resolve_soup_env
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.data.json_utils import parse_llm_json


def get_llm_client():
    """Retrieve an authenticated OpenAI-compatible client and target model from registry or environment."""
    connectors_path = "registry/connectors.json"
    api_key = None
    base_url = "https://api.openai.com/v1"
    model = None
    
    if os.path.exists(connectors_path):
        with open(connectors_path, "r", encoding="utf-8") as f:
            connectors = json.load(f)
            if connectors:
                conn = connectors[0]
                base_url = conn.get("base_url", base_url)
                model = conn.get("model")
                api_key = get_secret(conn["id"])
                print(f"✅ Utilizzo connettore: {conn['name']} (Modello: {model or 'gemini-1.5-flash'})", flush=True)

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("❌ Errore: Nessun connettore con API Key trovata. Configuralo nella UI (Connettori API) o esporta OPENAI_API_KEY.", flush=True)
        sys.exit(1)
        
    return OpenAI(api_key=api_key, base_url=base_url, timeout=90.0), model


def run_forge_pipeline(args):
    """Execute the full Forge pipeline (data generation + fine-tuning + registry)."""
    print(f"=== FORGE TRAINING PIPELINE: {args.name} (Method: {args.method.upper()}) ===", flush=True)

    print("PROGRESS: 10", flush=True)
    print("1. Caricamento connettore da registry...", flush=True)
    client, conn_model = get_llm_client()
    
    print("PROGRESS: 20", flush=True)
    clean_name = re.sub(r"[^\w\-]", "_", args.name.strip().lower())
    dataset_path = f"datasets/forge_{clean_name}.jsonl"
    
    chosen_model = getattr(args, "generator_model", None) or conn_model or "gemini-2.0-flash"
    candidates = [chosen_model]
    for m in ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gpt-4o-mini"]:
        if m not in candidates:
            candidates.append(m)
    
    if args.method == "lima":
        print(f"3. Esecuzione Pipeline LIMA ad Alta Qualità (Modello primario: {candidates[0]}, Fallback: {candidates[1:4]}, Target: {args.target_samples} campioni)...", flush=True)
        from aiforge.data.lima_pipeline import LIMAPipeline
        
        target_model = candidates[0]
        pipeline = LIMAPipeline(
            client=client,
            model_name=target_model,
            min_request_delay=2.0,
            max_retries=5,
            fallback_models=candidates[1:],
        )
        
        def lima_progress(pct, msg):
            scaled_pct = int(20 + (pct * 0.40))  # Scale LIMA generation to 20-60%
            print(f"PROGRESS: {scaled_pct}", flush=True)
            print(f"LIMA [{pct}%]: {msg}", flush=True)

        turn_type = getattr(args, "turn_type", "single_turn")
        res_lima = pipeline.run_pipeline(
            job_id=clean_name,
            main_topic=args.topic,
            target_samples=args.target_samples,
            batch_size=args.batch_size,
            output_final_path=dataset_path,
            turn_type=turn_type,
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
                print(f"⚠️ {target_model} non disponibile: {err}", flush=True)
                time.sleep(2)

        if not response:
            print("❌ Generazione fallita.", flush=True)
            sys.exit(1)

        print("PROGRESS: 40", flush=True)
        try:
            items = parse_llm_json(response.choices[0].message.content or "")
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
    
    soup_env = resolve_soup_env()
    from aiforge.core.resources import select_device
    requested_device = getattr(args, "device", None) or "auto"
    try:
        device_used = select_device(requested_device)
    except Exception as e:
        print(f"❌ Device non disponibile: {e}", flush=True)
        sys.exit(1)
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
        device=device_used,
        backend="soup",
        backend_args={
            "soup_env_path": soup_env,
            "chat_template": "chatml",
            "quantization": "4bit" if args.qlora else "none"
        }
    )

    print("PROGRESS: 70", flush=True)
    print(f"5. Avvio motore di training Soup (device: {device_used})...", flush=True)
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
    exp_id = registry.save_experiment(
        config=cfg,
        training_results={"history": res.get("history", {}), "soup_version": res.get("soup_version")},
        evaluation_results={},
        device_used=device_used,
        experiment_id=clean_name,
        checkpoint_path=adapter_path,
    )
    print("PROGRESS: 100", flush=True)
    print("🎉 Pipeline Forge Completata con Successo!", flush=True)


def run_lima_generate(args):
    """Generate curated synthetic dataset using LIMA without running training."""
    topic = args.topic
    target_samples = getattr(args, "target_samples", 100)
    batch_size = getattr(args, "batch_size", 20)
    output_path = getattr(args, "output", None)
    if not output_path:
        clean_topic = re.sub(r"[^\w\-]", "_", topic.strip().lower())[:30]
        output_path = f"datasets/lima_{clean_topic}.jsonl"
        
    print(f"=== LIMA SYNTHETIC DATASET GENERATION ===", flush=True)
    print(f"Topic: {topic}")
    print(f"Target Samples: {target_samples} (Micro-batch: {batch_size})")
    print(f"Output File: {output_path}")

    client, conn_model = get_llm_client()
    from aiforge.data.lima_pipeline import LIMAPipeline
    target_model = getattr(args, "model", None) or conn_model or "gemini-3.8-flash"
    
    pipeline = LIMAPipeline(
        client=client,
        model_name=target_model,
        min_request_delay=1.0,
        max_retries=4
    )
    
    def on_progress(pct, msg):
        print(f"[{pct}%] {msg}", flush=True)
        
    clean_job = re.sub(r"[^\w\-]", "_", os.path.basename(output_path).replace(".jsonl", ""))
    turn_type = getattr(args, "turn_type", "single_turn")
    result = pipeline.run_pipeline(
        job_id=clean_job,
        main_topic=topic,
        target_samples=target_samples,
        batch_size=batch_size,
        output_final_path=output_path,
        turn_type=turn_type,
        progress_cb=on_progress
    )
    print(f"✅ Completato! Dataset LIMA generato con successo: {result['curated_count']} campioni salvati in {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="AIFORGE Unified Forge Training Pipeline")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--model", required=True, help="Base model HuggingFace ID")
    parser.add_argument("--topic", required=True, help="Topic for synthetic dataset generation")
    parser.add_argument("--qlora", action="store_true", default=False, help="Enable 4-bit NF4 QLoRA quantization")
    parser.add_argument("--method", choices=["quick", "lima"], default="quick", help="Dataset generation strategy: 'quick' or 'lima'")
    parser.add_argument("--target-samples", type=int, default=100, help="Target curated samples count for LIMA")
    parser.add_argument("--batch-size", type=int, default=20, help="Micro-batch size for LIMA generation")
    parser.add_argument("--turn-type", choices=["single_turn", "multi_turn"], default="single_turn", help="Dialogue structure: 'single_turn' or 'multi_turn'")
    parser.add_argument("--generator-model", default=None, help="Teacher LLM model for data generation (e.g. gemini-1.5-flash)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device (auto: cuda -> mps -> cpu)")
    args = parser.parse_args()

    run_forge_pipeline(args)


if __name__ == "__main__":
    main()
