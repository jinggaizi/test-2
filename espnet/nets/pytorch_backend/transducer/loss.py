#!/usr/bin/env python3

"""Transducer loss module."""

import torch


class TransLoss(torch.nn.Module):
    """Transducer loss module.

    Args:
        trans_type (str): type of transducer implementation to calculate loss.
        blank_id (int): blank symbol id
    """

    def __init__(self, trans_type, lamb, joint_memory_reduction, blank_id):
        """Construct an TransLoss object."""
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if trans_type == "warp-transducer":
            from warprnnt_pytorch import RNNTLoss

            self.trans_loss = RNNTLoss(blank=blank_id)
        elif trans_type == "warp-rnnt":
            if device.type == "cuda":
                try:
                    from warp_rnnt import rnnt_loss

                    self.trans_loss = rnnt_loss
                except ImportError:
                    raise ImportError(
                        "warp-rnnt is not installed. Please re-setup"
                        " espnet or use 'warp-transducer'"
                    )
            else:
                raise ValueError("warp-rnnt is not supported in CPU mode")

        self.trans_type = trans_type
        self.blank_id = blank_id
        self.lamb = lamb
        self.joint_memory_reduction = joint_memory_reduction

    def forward(self, pred_pad, target, pred_len, target_len):
        """Compute path-aware regularization transducer loss.

        Args:
            pred_pad (torch.Tensor): Batch of predicted sequences
                (batch, maxlen_in, maxlen_out+1, odim)
            target (torch.Tensor): Batch of target sequences (batch, maxlen_out)
            pred_len (torch.Tensor): batch of lengths of predicted sequences (batch)
            target_len (torch.tensor): batch of lengths of target sequences (batch)

        Returns:
            loss (torch.Tensor): transducer loss

        """
        dtype = pred_pad.dtype
        if dtype != torch.float32:
            # warp-transducer and warp-rnnt only support float32
            pred_pad = pred_pad.to(dtype=torch.float32)
        if self.joint_memory_reduction:
            batch = target.size(0)
            loss = torch.zeros((1), dtype=torch.float32, device=pred_pad.device)
            _start = 0

            for b in range(batch):
                t = int(pred_len[b])
                u = int(target_len[b])
                t_u = t * (u + 1)

                if self.trans_type == "warp-rnnt":
                    log_probs = torch.log_softmax(
                        pred_pad[_start : (_start + t_u), :].view(1, t, (u + 1), -1),
                        dim=-1,
                    )
                loss += self.trans_loss(
                        log_probs,
                        target[b : (b + 1), :u],
                        pred_len[b].unsqueeze(0),
                        target_len[b].unsqueeze(0),
                        reduction="mean",
                        blank=self.blank_id,
                        gather=True,
                    )
                _start += t_u
            loss /= batch
        else:
            if self.trans_type == "warp-rnnt":
                log_probs = torch.log_softmax(pred_pad, dim=-1)
                loss = self.trans_loss(
                    log_probs,
                    target,
                    pred_len,
                    target_len,
                    reduction="mean",
                    blank=self.blank_id,
                    gather=True,
                    # fastemit_lambda=self.lamb,
                )
            else:
                loss = self.trans_loss(pred_pad, target, pred_len, target_len)
        loss = loss.to(dtype=dtype)
        
        return loss
