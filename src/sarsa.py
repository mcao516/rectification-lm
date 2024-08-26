# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class SARSA(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        gamma=1.0,
        q_max=0.0,
        q_min=-1.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.q_max = q_max
        self.q_min = q_min

    def _mask_loss(self, loss, actions):
        assert loss.shape == actions.shape
        pad_mask = actions.eq(self.ignore_index)
        loss.masked_fill_(pad_mask, 0.0)

    def forward(self, outputs, batch, reduce=True):
        """Compute TD Loss

        Args:
            logits (Tensor):        [bsz, tgt_len, vocab_size]
            tgt_logits (Tensor):    [bsz, tgt_len, vocab_size]
            actions (Tensor):       [bsz, tgt_len - 1]
            rewards (Tensor):       [bsz]
            seq_lens (Tensor):      [bsz]

        """
        _, (logits, tgt_logits, _) = outputs
        logits, tgt_logits = logits[0], tgt_logits[0]
        bsz = logits.shape[0]
        n_nonterminal = max(1, torch.sum(batch['input_ids'] != self.ignore_index))

        actions = batch['input_ids'][:, 1:]
        rewards = batch['rewards'].squeeze(-1).to(logits)
        seq_lens = torch.sum(batch['input_ids'] != self.ignore_index, dim=-1) - 1

        logits, tgt_logits = logits[:, :-1, :], tgt_logits[:, :-1, :]

        assert logits.dim() == actions.dim() + 1
        Q = logits.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [batch_size, tgt_len]

        # calculate TD target
        Q_backup = torch.zeros_like(Q)
        # Q_backup[:, :-1] = self.gamma * tgt_logits.max(-1)[0][:, 1:].detach()  # Q learning
        Q_backup[:, :-1] = self.gamma * tgt_logits.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)[:, 1:]
        Q_backup[torch.arange(bsz), seq_lens - 1] = rewards
        Q_backup = torch.clamp(Q_backup, max=self.q_max, min=self.q_min)

        loss = F.mse_loss(Q, Q_backup.detach(), reduction="none")
        self._mask_loss(loss, actions)

        if reduce:
            loss = loss.sum()

        return loss / n_nonterminal