import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


def compute_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False
) -> torch.Tensor:
    """
    Computes token log probabilities for target sequences where labels != -100.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        labels: Target label tokens [batch_size, seq_len] with -100 as padding/ignore
        average_log_prob: If True, normalizes by the sequence token count
        
    Returns:
        Tensor of per-example sum of log probabilities [batch_size]
    """
    # Shift logits and labels by 1 for causal autoregressive modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Log softmax across vocabulary dimension
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probabilities corresponding to actual label tokens
    # Mask invalid indices (-100) before gather
    loss_mask = (shift_labels != -100)
    safe_labels = shift_labels.clone()
    safe_labels[~loss_mask] = 0

    per_token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token_log_probs = per_token_log_probs * loss_mask.float()

    seq_log_probs = per_token_log_probs.sum(dim=-1)
    if average_log_prob:
        token_counts = loss_mask.sum(dim=-1).clamp(min=1.0)
        seq_log_probs = seq_log_probs / token_counts

    return seq_log_probs


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss.
    Reference: Rafailov et al. (NeurIPS 2023)
    
    Loss:
        L_DPO = -E [ log sigma( beta * ( log(pi/ref)_chosen - log(pi/ref)_rejected ) ) ]
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes DPO loss and preference alignment metrics.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = self.beta * (pi_logratios - ref_logratios)

        if self.label_smoothing > 0.0:
            losses = (
                -F.logsigmoid(logits) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)

        loss = losses.mean()

        # Compute implicit rewards and reward accuracy
        chosen_rewards = (self.beta * (policy_chosen_logps - reference_chosen_logps)).detach()
        rejected_rewards = (self.beta * (policy_rejected_logps - reference_rejected_logps)).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": reward_accuracies.mean().item()
        }

        return loss, metrics


class DPOBatchValidator:
    """Validates and processes preference tuning datasets."""

    @staticmethod
    def validate_sample(sample: Dict[str, Any]) -> bool:
        if not isinstance(sample, dict):
            return False
        prompt = sample.get("prompt")
        chosen = sample.get("chosen")
        rejected = sample.get("rejected")

        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            return False
        if not chosen or not isinstance(chosen, str) or not chosen.strip():
            return False
        if not rejected or not isinstance(rejected, str) or not rejected.strip():
            return False
        if chosen.strip() == rejected.strip():
            return False  # Degenerate pair (identical chosen and rejected)
        return True

    @classmethod
    def filter_preference_pairs(cls, pairs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        valid: List[Dict[str, str]] = []
        invalid_count = 0
        identical_count = 0

        for item in pairs:
            if not isinstance(item, dict):
                invalid_count += 1
                continue
            prompt = (item.get("prompt") or "").strip()
            chosen = (item.get("chosen") or "").strip()
            rejected = (item.get("rejected") or "").strip()

            if not prompt or not chosen or not rejected:
                invalid_count += 1
                continue
            if chosen == rejected:
                identical_count += 1
                continue

            valid.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        return valid, {
            "total": len(pairs),
            "valid": len(valid),
            "invalid_removed": invalid_count,
            "identical_removed": identical_count
        }
