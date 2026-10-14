import os
import json
import pytest
from unittest.mock import MagicMock
from aiforge.data.lima_pipeline import (
    LIMAQualityFilter,
    LIMAStagingManager,
    LIMAPipeline
)

def test_lima_quality_filter_similarity():
    text1 = "Scrivi una funzione C per allocare una matrice bidimensionale"
    text2 = "Scrivi una funzione C per allocare una matrice bidimensionale dinamica"
    text3 = "Spiega il protocollo TCP e il three-way handshake"

    sim_high = LIMAQualityFilter.jaccard_similarity(text1, text2)
    sim_low = LIMAQualityFilter.jaccard_similarity(text1, text3)

    assert sim_high > 0.70
    assert sim_low < 0.20


def test_lima_quality_filter_deduplication_and_length():
    samples = [
        # Valid sample 1
        {
            "instruction": "Come allocare memoria con malloc in C in modo sicuro?",
            "input": "",
            "output": "Per allocare memoria con malloc in modo sicuro, devi sempre controllare che il puntatore restituito non sia NULL prima di dereferenziarlo."
        },
        # Duplicate of sample 1
        {
            "instruction": "Come allocare memoria con malloc in C in modo sicuro e corretto?",
            "input": "",
            "output": "Usa malloc e controlla che il puntatore restituito non sia NULL."
        },
        # Too short output
        {
            "instruction": "Cos'è un puntatore void?",
            "input": "",
            "output": "Un puntatore generico."
        },
        # Malformed sample
        {
            "instruction": "",
            "input": "",
            "output": "Risposta senza domanda."
        },
        # Valid sample 2 (Distinct topic)
        {
            "instruction": "Come implementare un mutex POSIX per prevenire race conditions?",
            "input": "",
            "output": "Un mutex POSIX (pthread_mutex_t) viene inizializzato con pthread_mutex_init. Viene bloccato con pthread_mutex_lock e rilasciato con pthread_mutex_unlock nella sezione critica."
        }
    ]

    curated, stats = LIMAQualityFilter.filter_samples(samples, min_output_len=30, max_similarity_threshold=0.75)

    assert stats["total_input"] == 5
    assert stats["accepted"] == 2
    assert stats["duplicates_removed"] == 1
    assert stats["too_short_removed"] == 1
    assert stats["malformed_removed"] == 1
    assert len(curated) == 2


def test_lima_staging_manager_persistence(tmp_path):
    staging_dir = str(tmp_path / "staging")
    manager = LIMAStagingManager(staging_dir=staging_dir)
    job_id = "test_job_123"

    batch1 = [
        {"instruction": "Inst 1", "output": "Output spiegazione estesa per campione 1."},
        {"instruction": "Inst 2", "output": "Output spiegazione estesa per campione 2."}
    ]
    manager.append_batch(job_id, "Puntatori", batch1)

    # Resume check
    staged, completed = manager.load_progress(job_id)
    assert len(staged) == 2
    assert "Puntatori" in completed

    # Append second batch
    batch2 = [
        {"instruction": "Inst 3", "output": "Output spiegazione estesa per campione 3."}
    ]
    manager.append_batch(job_id, "Concorrenza", batch2)

    staged_after, completed_after = manager.load_progress(job_id)
    assert len(staged_after) == 3
    assert len(completed_after) == 2
    assert "Concorrenza" in completed_after


def test_lima_pipeline_mocked_end_to_end(tmp_path):
    mock_client = MagicMock()
    
    # Mock taxonomy response
    mock_taxonomy_resp = MagicMock()
    mock_taxonomy_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps([
            "Gestione Memoria Heap",
            "Puntatori e Aritmetica"
        ])))
    ]

    # Mock batch response
    mock_batch_resp = MagicMock()
    mock_batch_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps([
            {"instruction": "Domanda A sulla memoria", "input": "", "output": "Spiegazione approfondita e codice dettagliato per l'argomento A."},
            {"instruction": "Domanda B sui puntatori", "input": "", "output": "Spiegazione approfondita e codice dettagliato per l'argomento B."}
        ])))
    ]

    mock_client.chat.completions.create.side_effect = [
        mock_taxonomy_resp,
        mock_batch_resp,
        mock_batch_resp
    ]

    staging_dir = str(tmp_path / "staging")
    final_output = str(tmp_path / "final_lima.jsonl")

    pipeline = LIMAPipeline(
        client=mock_client,
        model_name="mock-model",
        staging_dir=staging_dir,
        min_request_delay=0.0
    )

    progress_reports = []
    def on_progress(pct, msg):
        progress_reports.append((pct, msg))

    res = pipeline.run_pipeline(
        job_id="test_run",
        main_topic="Programmazione C",
        target_samples=4,
        batch_size=2,
        output_final_path=final_output,
        progress_cb=on_progress
    )

    assert res["curated_count"] > 0
    assert os.path.exists(final_output)
    assert len(progress_reports) > 0
    assert progress_reports[-1][0] == 100


def test_lima_pipeline_retry_on_rate_limit():
    mock_client = MagicMock()
    
    mock_success_resp = MagicMock()
    mock_success_resp.choices = [MagicMock(message=MagicMock(content="[\"Sotto-argomento 1\"]"))]

    # Fail first with 429, then succeed
    mock_client.chat.completions.create.side_effect = [
        Exception("429 Too Many Requests: Rate limit exceeded"),
        mock_success_resp
    ]

    pipeline = LIMAPipeline(
        client=mock_client,
        model_name="mock-model",
        min_request_delay=0.0,
        max_retries=2
    )

    subtopics = pipeline.generate_taxonomy("Test Topic", target_subtopics=1)
    assert len(subtopics) == 1
    assert subtopics[0] == "Sotto-argomento 1"


def test_lima_staging_manager_path_traversal_prevention(tmp_path):
    staging_dir = str(tmp_path / "staging")
    manager = LIMAStagingManager(staging_dir=staging_dir)

    malicious_ids = [
        "../../etc/passwd",
        "../sensitive",
        "sub/dir/attack",
        "..\\windows\\system32",
        "",
        None
    ]

    for bad_id in malicious_ids:
        with pytest.raises(ValueError):
            manager.get_paths(bad_id)

