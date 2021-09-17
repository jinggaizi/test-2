# -*- coding: utf-8 -*-

"""Streaming related utility tools."""

import logging
import numpy as np
import torch
import random

import pdb


def chunk_level_mask(chunk_length, encoder_look_forward, padded_length, down_sampling):
    chunk_length //= down_sampling
    padded_length //= down_sampling
    chunk_num = padded_length // chunk_length
    true_box = torch.ones([chunk_length, chunk_length])
    assert chunk_num > 0
    if chunk_num == 1:
        return true_box, 1
    false_box = torch.zeros([chunk_length, chunk_length])
    ret = torch.zeros([1, chunk_num * chunk_length])
    for i in range(0, chunk_num):
        true_num = min(chunk_num, i + 1 + encoder_look_forward)
        false_num = chunk_num - true_num
        if false_num > 0:
            row = torch.cat([true_box.repeat(1, true_num), false_box.repeat(1, false_num)], 1)
        else:
            row = true_box.repeat(1, true_num)
        ret = torch.cat([ret, row], 0)
    return ret[1:], chunk_num


'''
def dynamic_chunk_level_mask(chunk_length, encoder_look_forward, padded_length, down_sampling, global_prob=0.25):
    if random.random() < global_prob:
        # get global context in global_prob
        return chunk_level_mask(padded_length, encoder_look_forward, padded_length, down_sampling)
    else:
        # use dynamic chunk
        least_chunk = 0
        most_chunk = chunk_length
        chunk_length = random.randint(least_chunk, most_chunk)
'''



def length_mask(lengths, maxlen):
    bs = int(len(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


# Wenet
def add_optional_chunk_mask(xs: torch.Tensor, masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int, static_chunk_size: int,
                            num_decoding_left_chunks: int):
    """ Apply optional mask for encoder.
    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if random.random() > 0.85:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 40 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks

    