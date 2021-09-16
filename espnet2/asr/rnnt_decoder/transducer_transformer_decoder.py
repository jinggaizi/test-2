# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import six
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transducer.transformer_decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask_limit
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.rnnt_decoder.abs_rnnt_decoder import AbsRNNTDecoder
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import get_activation
import logging
from espnet.nets.pytorch_backend.transducer.utils import check_state
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork

class BaseTransducerTransformerDecoder(AbsRNNTDecoder, BatchScorerInterface):
    """Base class of Transducer Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        blank: int = 0,
        joint_activation_type="tanh",
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
 
        self.blank = blank
        self.odim = vocab_size
        # Must set by the inheritance
        self.decoders = None
        self.joint_network = None
    def forward(
        self,
        hs_pad: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_mask: (batch, maxlen_out)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """

        tgt = self.embed(ys_in_pad)
        tgt, tgt_mask = self.decoders(
            tgt, ys_mask
        )
        if self.normalize_before:
            tgt = self.after_norm(tgt)

        h_enc = hs_pad.unsqueeze(2)
        h_dec = tgt.unsqueeze(1)

        z = self.joint_network(h_enc, h_dec)

        olens = tgt_mask.sum(1)
        return z, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask = decoder(x, tgt_mask, c)
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        return y, new_cache

    def score(self, ys, state, x):
        """Score."""
        #tgt_mask = to_device(self, subsequent_mask(len(ys)).unsqueeze(0))
        tgt_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        state = check_state(state, (ys.unsqueeze(0).size(1) - 1), self.blank)
        logp, state = self.forward_one_step(ys.unsqueeze(0), tgt_mask, cache=state)
        logp = torch.log_softmax(self.joint_network(x, logp[0]), dim=-1)
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, cache=batch_state)
        # logp = torch.log_softmax(self.joint(xs, logp), dim=-1)
        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def recognize(self, h, recog_args, target_left_mask=-1):
        """Greedy search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        hyp = {"score": 0.0, "yseq": [self.blank]}

        ys = to_device(self, torch.tensor(hyp["yseq"], dtype=torch.long)).unsqueeze(0)
        if target_left_mask > -1:
            ys_mask = to_device(self, subsequent_mask_limit(1, target_left_mask).unsqueeze(0))
        else:
            ys_mask = to_device(self, subsequent_mask(1).unsqueeze(0))
        y, c = self.forward_one_step(ys, ys_mask, None)

        for i, hi in enumerate(h):
            ytu = torch.log_softmax(self.joint_network(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp["yseq"].append(int(pred))
                hyp["score"] += float(logp)

                ys = to_device(self, torch.tensor(hyp["yseq"]).unsqueeze(0))
                if target_left_mask > -1:
                    ys_mask = to_device(
                        self, subsequent_mask_limit(len(hyp["yseq"]), target_left_mask).unsqueeze(0)
                    )
                else:
                    ys_mask = to_device(
                        self, subsequent_mask(len(hyp["yseq"])).unsqueeze(0)
                    )

                y, c = self.forward_one_step(ys, ys_mask, c)

        return [hyp]

    def recognize_beam(self, h, recog_args, rnnlm=None, target_left_mask=-1):
        """Beam search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language model module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        k_range = min(beam, self.odim)
        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        if rnnlm:
            kept_hyps = [
                {"score": 0.0, "yseq": [self.blank], "cache": None, "lm_state": None}
            ]
        else:
            kept_hyps = [{"score": 0.0, "yseq": [self.blank], "cache": None}]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                ys = to_device(self, torch.tensor(new_hyp["yseq"]).unsqueeze(0))
                if target_left_mask > -1:
                     ys_mask = to_device(
                        self, subsequent_mask_limit(len(new_hyp["yseq"]), target_left_mask).unsqueeze(0)
                    )
                else:
                    ys_mask = to_device(
                        self, subsequent_mask(len(new_hyp["yseq"])).unsqueeze(0)
                    )
                y, c = self.forward_one_step(ys, ys_mask, new_hyp["cache"])

                ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(
                        new_hyp["lm_state"], ys[:, -1]
                    )

                for k in six.moves.range(self.odim):
                    beam_hyp = {
                        "score": new_hyp["score"] + float(ytu[k]),
                        "yseq": new_hyp["yseq"][:],
                        "cache": new_hyp["cache"],
                    }

                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["yseq"].append(int(k))
                        beam_hyp["cache"] = c

                        if rnnlm:
                            beam_hyp["lm_state"] = rnnlm_state
                            beam_hyp["score"] += (
                                recog_args.lm_weight * rnnlm_scores[0][k]
                            )

                        hyps.append(beam_hyp)

                if len(kept_hyps) >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True)[
                :nbest
            ]

        return nbest_hyps

class TransducerTransformerDecoder(BaseTransducerTransformerDecoder):
    def __init__(
        self,
        jdim: int,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        blank: int = 0,
        joint_activation_type="tanh",
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            blank=blank,
            joint_activation_type=joint_activation_type,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                     attention_heads, attention_dim, self_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                 normalize_before,
                concat_after,
            ),
        )

        self.joint_network = JointNetwork(vocab_size, attention_dim, attention_dim, jdim, joint_activation_type)
