import random
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.attentions import initial_att
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet2.asr.rnnt_decoder.abs_rnnt_decoder import AbsRNNTDecoder
from espnet.nets.scorer_interface import BatchScorerInterface
import pdb

class TransducerRNNDecoder(AbsRNNTDecoder, BatchScorerInterface):
    """RNN-T Decoder module.

    Args:
        encoder_output_size (int): # encoder projection units
        vocab_size (int): dimension of outputs
        dtype (str): gru or lstm
        dlayers (int): # prediction layers
        dunits (int): # prediction units
        blank (int): blank symbol id
        embed_dim (init): dimension of embeddings
        joint_dim (int): dimension of joint space
        joint_activation_type (int): joint network activation
        dropout (float): dropout rate
        dropout_embed (float): embedding dropout rate

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dtype: str,
        dlayers: int,
        dunits: int,
        embed_dim: int,
        joint_dim: int,
        blank: int = 0,
        dropout: float = 0.0,
        dropout_embed: float = 0.0,
        joint_activation_type="tanh",
    ):
        """Transducer initializer."""
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=blank)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        if dtype == "lstm":
            dec_net = torch.nn.LSTMCell
        else:
            dec_net = torch.nn.GRUCell

        self.decoder = torch.nn.ModuleList([dec_net(embed_dim, dunits)])
        self.dropout_dec = torch.nn.ModuleList([torch.nn.Dropout(p=dropout)])

        for _ in range(1, dlayers):
            self.decoder += [dec_net(dunits, dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]

        self.joint_network = JointNetwork(
            vocab_size, encoder_output_size, dunits, joint_dim, joint_activation_type
        )

        self.dlayers = dlayers
        self.dunits = dunits
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.joint_dim = joint_dim
        self.vocab_size = vocab_size

        self.ignore_id = -1
        self.blank = blank

    def init_state(self, init_tensor):
        """Initialize decoder states.

        Args:
            init_tensor (torch.Tensor): batch of input features
                (B, emb_dim / dec_dim)

        Returns:
            (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])

        """
        dtype = init_tensor.dtype
        z_list = [
            to_device(init_tensor, torch.zeros(1, self.dunits)).to(
                dtype
            )
            for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(init_tensor, torch.zeros(1, self.dunits)).to(
                dtype
            )
            for _ in range(self.dlayers)
        ]

        return (z_list, c_list)
    
    def zero_state(self, init_tensor):
        """Initialize decoder states.

        Args:
            init_tensor (torch.Tensor): batch of input features
                (B, emb_dim / dec_dim)

        Returns:
            (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])

        """
        dtype = init_tensor.dtype
        z_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits)).to(
                dtype
            )
            for _ in range(self.dlayers)
        ]
        c_list = [
            to_device(init_tensor, torch.zeros(init_tensor.size(0), self.dunits)).to(
                dtype
            )
            for _ in range(self.dlayers)
        ]

        return (z_list, c_list)

    def rnn_forward(self, ey, state):
        """RNN forward.

        Args:
            ey (torch.Tensor): batch of input features (B, emb_dim)
            state (tuple): batch of decoder states
                (L x (B, dec_dim), L x (B, dec_dim))

        Returns:
            y (torch.Tensor): batch of output features (B, dec_dim)
            (tuple): batch of decoder states
                (L x (B, dec_dim), L x (B, dec_dim))

        """
        z_prev, c_prev = state
        z_list, c_list = self.zero_state(ey)

        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))

            for i in range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), (z_prev[i], c_prev[i])
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])

            for i in range(1, self.dlayers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        y = self.dropout_dec[-1](z_list[-1])

        return y, (z_list, c_list)

    def forward(self, hs_pad, ys_in_pad, hlens=None):
        """Forward function for transducer.

        Args:
            hs_pad (torch.Tensor):
                batch of padded hidden state sequences (B, Tmax, D)
            ys_in_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax+1)

        Returns:
            z (torch.Tensor): output (B, T, U, vocab_size)

        """
        olength = ys_in_pad.size(1)

        state = self.zero_state(hs_pad)
        eys = self.dropout_embed(self.embed(ys_in_pad))

        z_all = []
        for i in range(olength):
            y, state = self.rnn_forward(eys[:, i, :], state)
            z_all.append(y)

        h_enc = hs_pad.unsqueeze(2)

        h_dec = torch.stack(z_all, dim=1)
        h_dec = h_dec.unsqueeze(1)

        z = self.joint_network(h_enc, h_dec)

        return z, ys_in_pad

    def score(self, hyp, state, x):
        """Forward one step.

        Args:
            hyp (dataclass): hypothesis
            cache (dict): states cache

        Returns:
            y (torch.Tensor): decoder outputs (1, dec_dim)
            state (tuple): decoder states
                ([L x (1, dec_dim)], [L x (1, dec_dim)]),
            (torch.Tensor): token id for LM (1)

        """
        vy = to_device(self, torch.full((1, 1), hyp.yseq[-1], dtype=torch.long))
        ey = self.embed(vy)

        y, state = self.rnn_forward(ey[0], state)

        return y, state

    # def batch_score(self, hyps, batch_states, cache, init_tensor=None):
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Forward batch one step.

        Args:
            hyps (list): batch of hypotheses
            batch_states (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])
            cache (dict): states cache

        Returns:
            batch_y (torch.Tensor): decoder output (B, dec_dim)
            batch_states (tuple): batch of decoder states
                ([L x (B, dec_dim)], [L x (B, dec_dim)])
            lm_tokens (torch.Tensor): batch of token ids for LM (B)

        """
        n_batch = len(ys)
        n_layers = len(self.decoder)
        # pdb.set_trace()
        # if states[0] is None:
        #     batch_state = None
        # else:
        #     # transpose state of [batch, layer] into [layer, batch]

        #     batch_state = [
        #         torch.stack([states[b][i] for b in range(n_batch)])
        #         for i in range(n_layers)
        #     ]
        ey = self.embed(ys)
        y, states = self.rnn_forward(ey[:,-1:,:].squeeze(0), states[0])
        # state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return y, [states]

    # def select_state(self, batch_states, idx):
    #     """Get decoder state from batch of states, for given id.

    #     Args:
    #         batch_states (tuple): batch of decoder states
    #             ([L x (B, dec_dim)], [L x (B, dec_dim)])
    #         idx (int): index to extract state from batch of states

    #     Returns:
    #         (tuple): decoder states for given id
    #             ([L x (1, dec_dim)], [L x (1, dec_dim)])

    #     """
    #     z_list = [batch_states[0][layer][idx] for layer in range(self.dlayers)]
    #     c_list = [batch_states[1][layer][idx] for layer in range(self.dlayers)]

    #     return (z_list, c_list)

    # def create_batch_states(self, batch_states, l_states, l_tokens=None):
    #     """Create batch of decoder states.

    #     Args:
    #         batch_states (tuple): batch of decoder states
    #            ([L x (B, dec_dim)], [L x (B, dec_dim)])
    #         l_states (list): list of decoder states
    #             [B x ([L x (1, dec_dim)], [L x (1, dec_dim)])]

    #     Returns:
    #         batch_states (tuple): batch of decoder states
    #             ([L x (B, dec_dim)], [L x (B, dec_dim)])

    #     """
    #     for layer in range(self.dlayers):
    #         batch_states[0][layer] = torch.stack([s[0][layer] for s in l_states])
    #         batch_states[1][layer] = torch.stack([s[1][layer] for s in l_states])

    #     return batch_states

    def recognize(self, h, recog_args):
        """Greedy search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        state = self.zero_state(h.unsqueeze(0))
        ey = to_device(self, torch.zeros((1, self.embed_dim)))

        hyp = {"score": 0.0, "yseq": [self.blank]}

        y, state = self.rnn_forward(ey, state)

        for hi in h:
            ytu = torch.log_softmax(self.joint_network(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp["yseq"].append(int(pred))
                hyp["score"] += float(logp)

                eys = to_device(
                    self, torch.full((1, 1), hyp["yseq"][-1], dtype=torch.long)
                )
                ey = self.dropout_embed(self.embed(eys))

                y, state = self.rnn_forward(ey[0], state)

        return [hyp]

    def recognize_beam(self, h, recog_args, rnnlm=None):
        """Beam search implementation.

        Args:
            h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        k_range = min(beam, self.vocab_size)
        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        state = self.zero_state(h.unsqueeze(0))
        eys = to_device(self, torch.zeros((1, self.embed_dim)))

        _, state = self.rnn_forward(eys, None)

        if rnnlm:
            kept_hyps = [
                {
                    "score": 0.0,
                    "yseq": [self.blank],
                    "state": state,
                    "lm_state": None,
                }
            ]
        else:
            kept_hyps = [
                {"score": 0.0, "yseq": [self.blank], "state": state}
            ]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                vy = to_device(
                    self, torch.full((1, 1), new_hyp["yseq"][-1], dtype=torch.long)
                )
                ey = self.dropout_embed(self.embed(vy))

                y, state = self.rnn_forward(
                    ey[0], new_hyp["state"]
                )

                ytu = torch.log_softmax(self.joint_network(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(
                        new_hyp["lm_state"], vy[0]
                    )

                for k in range(self.vocab_size):
                    beam_hyp = {
                        "score": new_hyp["score"] + float(ytu[k]),
                        "yseq": new_hyp["yseq"][:],
                        "state" : new_hyp["state"],
                    }
                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["state"] = state
                        beam_hyp["yseq"].append(int(k))

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