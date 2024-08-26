# coding=utf-8

from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def make_head(n_embd: int, out: int, drop_prob: float=0.1):
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(n_embd * 2, out),
    )


class ILQLHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        n_vs: int = 1,
        alpha: float = 0.1,
        drop_prob: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_vs = n_vs
        self.alpha = alpha

        # self.v_head = make_head(self.hidden_size, 1)
        self.q_heads = nn.ModuleList(
            make_head(self.hidden_size, self.vocab_size, drop_prob=drop_prob) for _ in range(n_vs)
        )
        self.target_q_heads = nn.ModuleList(deepcopy(q_head) for q_head in self.q_heads)

        for q_head in self.target_q_heads:
            q_head.requires_grad_(False)

    def forward(
        self,
        hs: torch.Tensor,
    ):
        """
        Args:
            hs (torch.Tensor): [bsz, seq_len, hidden_size]

        Returns:
            qs (torch.Tensor): [bsz, seq_len, feature_size]
            target_qs (torch.Tensor): [bsz, seq_len, feature_size]
        """
        qs = tuple(q_head(hs) for q_head in self.q_heads)
        target_qs = tuple(q_head(hs) for q_head in self.target_q_heads)
        # vs = self.v_head(hs)
        assert len(qs) == 1, "the current implementation only supports n_qs == 1"

        return qs, target_qs, None

    def _sync_target_q_heads(self, alpha):
        for target_q_head, q_head in zip(self.target_q_heads, self.q_heads):
            for target_param, copy_param in zip(
                target_q_head.parameters(), q_head.parameters()
            ):
                target_param.data.copy_(
                    (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
                )

    def sync_target_q_heads(self):
        self._sync_target_q_heads(self.alpha)


class CausalLMWithValueHeads(GPT2LMHeadModel):
    """This is a wrapper around huggingface AutoModelForCausalLM with two additional scalar heads"""

    def __init__(self, config, alpha=0.1):
        super().__init__(config)

        self.alpha = alpha
        self.n_embd = self.transformer.config.n_embd
        self.vocab_size = self.transformer.config.vocab_size

        self.ilql_heads = ILQLHeads(
            self.n_embd, self.vocab_size, alpha=self.alpha, drop_prob=0.0
        )

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
    ):
        """
        Returns:
            qs (Tensor): [bsz, seq_len, vocab_size]
            target_qs (Tensor): [bsz, seq_len, vocab_size]

        """
        out = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = out[0]  # [bsz, seq_len, hidden_size]
        qs, target_qs, vs = self.ilql_heads(hidden_states)

        return None, qs, target_qs, vs

    @property
    def dummy_inputs(self):
        return {"input_ids": torch.ones(1, 1, device=self.gpt.device, dtype=torch.long)}

    @property
    def device(self):
        return self.gpt.device


class CausalLMWithValueHeadsInference(CausalLMWithValueHeads):
    """This is a wrapper around CausalLMWithValueHeadsInference for inference"""

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # lm_logits_for_test = self.lm_head(hidden_states)
        with torch.no_grad():
            lm_logits, _, _ = self.ilql_heads(hidden_states)
            if type(lm_logits) == tuple:
                lm_logits = lm_logits[0]
        # assert lm_logits.shape == lm_logits_for_test.shape, \
        #     "{} - {}".format(lm_logits.shape, lm_logits_for_test.shape)

        loss = None
        if labels is not None:
            raise NotImplementedError

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )