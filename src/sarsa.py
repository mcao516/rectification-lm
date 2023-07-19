# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class SARSA(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        gamma=1.0
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def _mask_loss(self, loss, actions):
        assert loss.shape == actions.shape
        pad_mask = actions.eq(self.ignore_index)
        loss.masked_fill_(pad_mask, 0.0)

    def forward(self, logits, tgt_logits, actions, rewards, seq_lens, reduce=True):
        """Compute DQN loss for given samples.

        Args:
            logits (Tensor): [batch_size, tgt_len, vocab_size]
            tgt_logits (Tensor): [batch_size, tgt_len, vocab_size]
            actions (Tensor): [batch_size, tgt_len]
            rewards (Tensor): [batch_size]
            seq_lens (Tensor): [batch_size]

        """
        batch_size = logits.shape[0]

        assert logits.dim() == actions.dim() + 1
        Q = logits.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [batch_size, tgt_len]

        # calculate TD target
        Q_backup = torch.zeros_like(Q)
        # Q_backup[:, :-1] = self.gamma * tgt_logits.max(-1)[0][:, 1:].detach()  # Q learning
        Q_backup[:, :-1] = self.gamma * tgt_logits.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)[:, 1:]
        Q_backup[torch.arange(batch_size), seq_lens - 1] = rewards
        Q_backup = torch.clamp(Q_backup, max=0.0, min=-1.0)

        loss = F.mse_loss(Q, Q_backup.detach(), reduction="none")
        self._mask_loss(loss, actions)

        if reduce:
            loss = loss.sum()

        return loss