"""
AIFORGE LIMA Pipeline (Less Is More for Alignment).

Provides a structured, high-quality synthetic data generation engine with:
1. Topic Tree Taxonomy Decomposition (sub-topics tree to avoid monolithic prompt exhaustion)
2. Micro-Batch Generation with Adaptive Rate Limiting & Exponential Backoff
3. Persistent Checkpoint & Staging with Resume capability
4. Local Zero-Cost Quality Filtering (Deduplication via Jaccard similarity and Completeness validation)
"""

import os
import json
import time
import math
import random
import re
from typing import List, Dict, Any, Optional, Tuple, Callable


class LIMAQualityFilter:
    """
    Local heuristic & semantic quality filter to enforce LIMA standards:
    - Eliminates duplicate or near-duplicate instructions.
    - Eliminates truncated, empty, or degenerate outputs.
    - Validates minimum structure and formatting.
    Runs entirely on the local CPU without consuming LLM API calls.
    """

    @staticmethod
    def tokenize_words(text: str) -> set:
        clean = re.sub(r"[^\w\s]", "", text.lower())
        return set(clean.split())

    @classmethod
    def jaccard_similarity(cls, text1: str, text2: str) -> float:
        set1 = cls.tokenize_words(text1)
        set2 = cls.tokenize_words(text2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / float(union) if union > 0 else 0.0

    @classmethod
    def filter_samples(
        cls,
        samples: List[Dict[str, str]],
        min_output_len: int = 40,
        max_similarity_threshold: float = 0.80
    ) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        """
        Filters a list of instruction-tuning samples.
        """
        accepted: List[Dict[str, str]] = []
        seen_instructions: List[str] = []
        
        duplicates_count = 0
        too_short_count = 0
        malformed_count = 0

        for item in samples:
            if "messages" in item or "turns" in item:
                # Multi-turn conversational sample
                msgs = item.get("messages") or item.get("turns") or []
                if not isinstance(msgs, list) or len(msgs) < 2:
                    malformed_count += 1
                    continue

                is_valid = True
                user_first = ""
                total_assistant_len = 0
                for turn in msgs:
                    if not isinstance(turn, dict) or not turn.get("content"):
                        is_valid = False
                        break
                    role = turn.get("role", "")
                    content = str(turn.get("content", "")).strip()
                    if role == "user" and not user_first:
                        user_first = content
                    if role == "assistant":
                        total_assistant_len += len(content)

                if not is_valid or not user_first:
                    malformed_count += 1
                    continue

                if total_assistant_len < min_output_len:
                    too_short_count += 1
                    continue

                # Check duplicate against previous prompts
                is_duplicate = False
                for prev_inst in seen_instructions:
                    sim = cls.jaccard_similarity(user_first, prev_inst)
                    if sim >= max_similarity_threshold:
                        is_duplicate = True
                        break

                if is_duplicate:
                    duplicates_count += 1
                    continue

                accepted.append(item)
                seen_instructions.append(user_first)
            else:
                # Single-turn instruction sample
                instruction = (item.get("instruction") or "").strip()
                output = (item.get("output") or "").strip()

                if not instruction or not output:
                    malformed_count += 1
                    continue

                if len(output) < min_output_len:
                    too_short_count += 1
                    continue

                # Check near-duplicate against already accepted instructions
                is_duplicate = False
                for prev_inst in seen_instructions:
                    sim = cls.jaccard_similarity(instruction, prev_inst)
                    if sim >= max_similarity_threshold:
                        is_duplicate = True
                        break

                if is_duplicate:
                    duplicates_count += 1
                    continue

                accepted.append({
                    "instruction": instruction,
                    "input": (item.get("input") or "").strip(),
                    "output": output
                })
                seen_instructions.append(instruction)

        stats = {
            "total_input": len(samples),
            "accepted": len(accepted),
            "duplicates_removed": duplicates_count,
            "too_short_removed": too_short_count,
            "malformed_removed": malformed_count
        }

        return accepted, stats


class LIMAStagingManager:
    """
    Manages persistent staging files and metadata to allow graceful resume
    if a network disconnection or rate limit occurs.
    """

    def __init__(self, staging_dir: str = "datasets/staging"):
        self.staging_dir = staging_dir
        os.makedirs(self.staging_dir, exist_ok=True)

    def _validate_job_id(self, job_id: str) -> None:
        if not job_id or not isinstance(job_id, str):
            raise ValueError("Invalid job_id: must be a non-empty string.")
        if ".." in job_id or "/" in job_id or "\\" in job_id:
            raise ValueError(f"Path traversal detected in job_id: {job_id}")

    def get_paths(self, job_id: str) -> Tuple[str, str]:
        self._validate_job_id(job_id)
        jsonl_path = os.path.join(self.staging_dir, f"{job_id}.jsonl")
        meta_path = os.path.join(self.staging_dir, f"{job_id}.meta.json")
        return jsonl_path, meta_path

    def load_progress(self, job_id: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Loads existing staged samples and completed subtopics.
        """
        jsonl_path, meta_path = self.get_paths(job_id)
        samples = []
        completed_subtopics = []

        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except Exception:
                            pass

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    completed_subtopics = meta.get("completed_subtopics", [])
            except Exception:
                pass

        return samples, completed_subtopics

    def append_batch(self, job_id: str, subtopic: str, new_samples: List[Dict[str, str]]) -> int:
        """
        Appends new valid samples to the staging file and updates completed subtopics metadata.
        """
        jsonl_path, meta_path = self.get_paths(job_id)
        
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for s in new_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        _, completed = self.load_progress(job_id)
        if subtopic not in completed:
            completed.append(subtopic)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "job_id": job_id,
                "updated_at": time.time(),
                "completed_subtopics": completed
            }, f, indent=2)

        return len(new_samples)


class LIMAPipeline:
    """
    Main Orchestrator for the LIMA synthetic data generation pipeline.
    """

    def __init__(
        self,
        client: Any,
        model_name: str = "gemini-3.8-flash",
        staging_dir: str = "datasets/staging",
        min_request_delay: float = 1.0,
        max_retries: int = 4
    ):
        self.client = client
        self.model_name = model_name
        self.staging = LIMAStagingManager(staging_dir)
        self.min_request_delay = min_request_delay
        self.max_retries = max_retries

    def generate_taxonomy(self, main_topic: str, target_subtopics: int = 15) -> List[str]:
        """
        Step 1: Decomposes the main topic into a comprehensive taxonomy tree of sub-topics.
        Prevents prompt exhaustion and redundancy.
        """
        prompt = f"""Dato il seguente argomento principale: "{main_topic}", crea una tassonomia di esattamente {target_subtopics} sotto-argomenti verticali, distinti e complementari, che coprano aspetti teorici, pratici, architetturali e casi limite (corner-cases).
Rispondi ESCLUSIVAMENTE con un array JSON di stringhe. Esempio:
["Sotto-argomento 1", "Sotto-argomento 2"]"""

        response = self._call_llm_with_retry(prompt)
        text = response.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        text = text.strip()

        try:
            subtopics = json.loads(text)
            if isinstance(subtopics, list) and len(subtopics) > 0:
                return [str(s).strip() for s in subtopics if str(s).strip()]
        except Exception:
            pass

        # Fallback heuristic taxonomy if parsing fails
        return [f"{main_topic} - Modulo {i+1}" for i in range(target_subtopics)]

    def generate_micro_batch(self, subtopic: str, count: int = 20, turn_type: str = "single_turn") -> List[Dict[str, Any]]:
        """
        Step 2: Generates a small micro-batch of structured examples for a specific sub-topic.
        Supports 'single_turn' (instruction/output) and 'multi_turn' (conversational dialogues).
        Chunks large requests into micro-batches of max 5 samples to prevent token truncation and API timeouts.
        """
        chunk_size = 5
        total_valid: List[Dict[str, Any]] = []
        remaining = max(1, count)

        while remaining > 0:
            current_chunk = min(remaining, chunk_size)
            if turn_type == "multi_turn":
                prompt = f"""Crea {current_chunk} conversazioni didattiche multi-turno di alta qualità per il seguente sotto-argomento:
"{subtopic}"

Ogni conversazione deve avere almeno 2 o 3 turni di interazione (domanda iniziale, risposta approfondita, domanda di follow-up o caso limite, e risposta finale con codice o dettagli tecnici).
Rispondi ESCLUSIVAMENTE con un array JSON valido, senza markdown.
Formato:
[
  {{
    "messages": [
      {{"role": "user", "content": "Domanda o problema iniziale"}},
      {{"role": "assistant", "content": "Spiegazione esaustiva iniziale"}},
      {{"role": "user", "content": "Approfondimento, corner-case o richiesta di ottimizzazione"}},
      {{"role": "assistant", "content": "Soluzione finale avanzata e dettagliata"}}
    ]
  }}
]"""
            else:
                prompt = f"""Crea {current_chunk} esempi didattici avanzati di alta qualità secondo la metodologia LIMA (istruzione chiara, non ambigua, con risposta completa, approfondita e priva di errori) per il seguente sotto-argomento:
"{subtopic}"

Rispondi ESCLUSIVAMENTE con un array JSON valido. Nessun markdown o testo introduttivo.
Formato:
[
  {{
    "instruction": "Domanda o compito specifico",
    "input": "",
    "output": "Spiegazione esaustiva e codice o soluzione dettagliata"
  }}
]"""

            try:
                response = self._call_llm_with_retry(prompt)
                text = response.strip()
                if text.startswith("```json"): text = text[7:]
                if text.startswith("```"): text = text[3:]
                if text.endswith("```"): text = text[:-3]
                text = text.strip()

                items = json.loads(text)
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        if turn_type == "multi_turn" and ("messages" in it or "turns" in it):
                            total_valid.append(it)
                        elif turn_type != "multi_turn" and "instruction" in it and "output" in it:
                            total_valid.append(it)
            except Exception as e:
                print(f"⚠️ Errore nel parsing del sotto-blocco JSON: {e}", flush=True)

            remaining -= current_chunk

        return total_valid

    def _call_llm_with_retry(self, prompt: str) -> str:
        """
        Calls the LLM client with adaptive exponential backoff and jitter to protect against 429 and rate limits.
        """
        retries = 0
        backoff = 2.0

        while retries <= self.max_retries:
            try:
                time.sleep(self.min_request_delay)
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return res.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "quota" in err_str or "overloaded" in err_str
                retries += 1
                if retries > self.max_retries:
                    print(f"❌ Errore critico LLM: superato il limite di {self.max_retries} tentativi ({e})", flush=True)
                    raise RuntimeError(f"Chiamata LLM fallita dopo {self.max_retries} tentativi: {e}")

                sleep_time = backoff + random.uniform(0.5, 1.5)
                print(f"⏳ Connessione o Quota ({retries}/{self.max_retries}): attesa {sleep_time:.1f}s prima di riprovare...", flush=True)
                time.sleep(sleep_time)
                backoff *= 2.0

        return ""

    def run_pipeline(
        self,
        job_id: str,
        main_topic: str,
        target_samples: int = 100,
        batch_size: int = 20,
        output_final_path: Optional[str] = None,
        turn_type: str = "single_turn",
        progress_cb: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end LIMA pipeline:
        Taxonomy -> Micro-Batch Staging -> Quality Filtering -> Final Dataset.
        """
        def report(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        report(5, "Inizializzazione pipeline LIMA...")
        
        # Check existing progress for resume
        staged_samples, completed_subtopics = self.staging.load_progress(job_id)
        report(10, f"Ripristino staging: {len(staged_samples)} campioni già generati.")

        needed_samples = max(0, target_samples - len(staged_samples))
        num_subtopics_needed = math.ceil(needed_samples / float(batch_size))
        
        if needed_samples > 0:
            try:
                report(15, f"Fase 1: Generazione albero tassonomico per '{main_topic}'...")
                taxonomy = self.generate_taxonomy(main_topic, target_subtopics=max(5, num_subtopics_needed))
                
                # Filter out already completed subtopics
                pending_subtopics = [s for s in taxonomy if s not in completed_subtopics]
                total_steps = len(pending_subtopics)

                report(25, f"Tassonomia pronta con {len(taxonomy)} sotto-argomenti. Inizio micro-batch...")

                for i, subtopic in enumerate(pending_subtopics):
                    if len(staged_samples) >= target_samples:
                        break

                    step_pct = int(25 + ((i + 1) / max(1, total_steps)) * 50)
                    report(step_pct, f"Generazione micro-batch ({i+1}/{total_steps}): {subtopic}...")

                    batch_items = self.generate_micro_batch(subtopic, count=batch_size, turn_type=turn_type)
                    if batch_items:
                        self.staging.append_batch(job_id, subtopic, batch_items)
                        staged_samples.extend(batch_items)
                        report(step_pct, f"Accreditati {len(batch_items)} campioni in staging ({len(staged_samples)}/{target_samples}).")
            except Exception as e:
                if len(staged_samples) >= 15:
                    report(75, f"⚠️ Quota API esaurita ({e}). Procedo con il salvataggio e addestramento dei {len(staged_samples)} campioni già presenti nello staging!")
                else:
                    raise

        report(80, "Fase 3: Applicazione LIMA Quality Gate (Deduplicazione e Sintassi in Locale)...")
        
        # Quality filter
        curated_samples, filter_stats = LIMAQualityFilter.filter_samples(
            staged_samples,
            min_output_len=40,
            max_similarity_threshold=0.80
        )

        final_path = output_final_path or f"datasets/lima_{job_id}.jsonl"
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        
        with open(final_path, "w", encoding="utf-8") as f:
            for sample in curated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        report(100, f"Pipeline LIMA completata! {len(curated_samples)} campioni curati salvati in {final_path}.")

        return {
            "job_id": job_id,
            "final_dataset_path": final_path,
            "raw_staged_count": len(staged_samples),
            "curated_count": len(curated_samples),
            "filter_stats": filter_stats
        }
