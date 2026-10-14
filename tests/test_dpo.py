import pytest
import torch
from aiforge.llm.dpo import (
    compute_sequence_log_probs,
    DPOLoss,
    DPOBatchValidator
)


def test_compute_sequence_log_probs_masking():
    # Batch size 2, seq len 4, vocab size 5
    logits = torch.randn(2, 4, 5, requires_grad=True)
    labels = torch.tensor([
        [1, 2, 3, 4],
        [1, 2, -100, -100]  # Second token is masked at the end
    ])

    log_probs = compute_sequence_log_probs(logits, labels, average_log_prob=False)
    assert log_probs.shape == (2,)
    assert not torch.isnan(log_probs).any()

    # Backprop test
    loss = log_probs.sum()
    loss.backward()
    assert logits.grad is not None


def test_dpo_loss_preference_ranking():
    dpo_criterion = DPOLoss(beta=0.1)

    # Scenario 1: Model strongly prefers chosen over rejected
    policy_chosen = torch.tensor([-1.0, -1.2])
    policy_rejected = torch.tensor([-5.0, -4.5])
    ref_chosen = torch.tensor([-3.0, -3.0])
    ref_rejected = torch.tensor([-3.0, -3.0])

    loss, metrics = dpo_criterion(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert metrics["accuracy"] == 1.0
    assert metrics["reward_margin"] > 0
    assert loss.item() < 0.693  # -log(0.5) ~ 0.693

    # Scenario 2: Model prefers rejected over chosen (misaligned)
    loss_bad, metrics_bad = dpo_criterion(policy_rejected, policy_chosen, ref_chosen, ref_rejected)
    assert metrics_bad["accuracy"] == 0.0
    assert metrics_bad["reward_margin"] < 0
    assert loss_bad.item() > loss.item()


def test_dpo_loss_invalid_beta():
    with pytest.raises(ValueError, match="beta must be positive"):
        DPOLoss(beta=-0.1)


def test_dpo_loss_label_smoothing():
    dpo_standard = DPOLoss(beta=0.1, label_smoothing=0.0)
    dpo_smoothed = DPOLoss(beta=0.1, label_smoothing=0.1)

    p_c = torch.tensor([-1.0])
    p_r = torch.tensor([-5.0])
    r_c = torch.tensor([-3.0])
    r_r = torch.tensor([-3.0])

    l_std, _ = dpo_standard(p_c, p_r, r_c, r_r)
    l_smooth, _ = dpo_smoothed(p_c, p_r, r_c, r_r)

    assert l_smooth.item() > 0
    assert l_std.item() != l_smooth.item()


def test_dpo_batch_validator():
    pairs = [
        # Valid
        {"prompt": "Write a quicksort in C", "chosen": "void quicksort(...){...}", "rejected": "I do not know"},
        # Missing rejected
        {"prompt": "Test", "chosen": "Ok", "rejected": ""},
        # Identical chosen & rejected
        {"prompt": "Test identical", "chosen": "Same text", "rejected": "Same text"},
        # Malformed non-dict
        "not a dict"
    ]

    valid, stats = DPOBatchValidator.filter_preference_pairs(pairs)
    assert len(valid) == 1
    assert stats["valid"] == 1
    assert stats["invalid_removed"] == 2
    assert stats["identical_removed"] == 1
    assert valid[0]["prompt"] == "Write a quicksort in C"
