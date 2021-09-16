# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import Optional
from typing import Tuple

import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.asr.streaming.subsampling import Conv2dSubsampling
from espnet2.asr.streaming.subsampling import Conv2dSubsampling6
from espnet2.asr.streaming.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
# from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling_Streaming
from espnet2.asr.encoder.abs_encoder import AbsEncoder

from espnet2.asr.streaming.mask import chunk_level_mask
from espnet2.asr.streaming.mask import add_optional_chunk_mask
# from espnet2.asr.streaming.convolution import CasualConvolutionModule
from espnet2.asr.streaming.encoder_layer import MultiMaskEncoderLayer
from espnet2.asr.streaming.attention import MultiMaskRelPositionMultiHeadedAttention
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer

import pdb
import math
import random


class ConformerStreamingCascadedEncoder(AbsEncoder):
    """Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimention of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_causal_blocks: int = 6,
        num_non_causal_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 31,
        zero_triu: bool = False,
        padding_idx: int = -1,
        causal: bool = True,
        chunk_mode: str = 'dynamic',
        chunk_length: int = 40,
        decoding_chunk_length: int = 40,
        global_prob: float = 0.0,
        non_causal: bool = False,
        causal_weight: float = 0.3,
        decoding: bool = False,
        simulate_streaming: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.chunk_mode = chunk_mode

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn" or selfattention_layer_type == "mm_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        elif selfattention_layer_type == "mm_rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = MultiMaskRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_causal_args = (output_size, cnn_module_kernel, activation, causal)
        convolution_layer_non_causal_args = (output_size, cnn_module_kernel, activation)

        self.causal_encoders = repeat(
            num_causal_blocks,
            lambda lnum: MultiMaskEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_causal_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.non_causal_encoders = repeat(
            num_non_causal_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                #encoder_selfattn_layer(*encoder_selfattn_layer_args),
                LegacyRelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_non_causal_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
         )

        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        # streaming related
        self.chunk_length = chunk_length
        self.global_prob = global_prob
        self.decoding_chunk_length = decoding_chunk_length
        self.decoding = decoding
        self.simulate_streaming = simulate_streaming
        self.non_causal = non_causal
        self.causal_weight = causal_weight
        #print(self.decoding)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        Emask = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        # Emask_ = Emask

        # TODO: make padding xs and generate Cmask in a function
        if self.chunk_mode == 'fix':
            bs, tmax, idim = xs_pad.shape
            to_pad = self.chunk_length * (math.ceil(tmax/self.chunk_length)) - tmax + 4

            ilens += to_pad

            xs_pad = torch.cat((xs_pad, torch.zeros([bs, to_pad, idim]).to(xs_pad.device)), 1)
            tmax_ = tmax + to_pad - 4

            Cmask, num_chunks = chunk_level_mask(chunk_length=self.chunk_length, encoder_look_forward=0, padded_length=tmax_, down_sampling=4)

            Emask_pad = torch.ones([bs, 1, to_pad]).eq(0).to(xs_pad.device) # generate False block and add it to Emask
            Emask = torch.cat([Emask, Emask_pad], dim=-1)
            # pdb.set_trace()
        elif self.chunk_mode == 'dynamic-u2':
            Emask = add_optional_chunk_mask(xs_pad, Emask, True, False, 0, -1, -1)
            Cmask = None
        elif self.chunk_mode == 'dynamic':
            # self.decoding=True
            if self.decoding is False:
                if random.random() < self.global_prob:
                    Cmask = None
                else:
                    l_scale = 0.2
                    r_scale = 1.2
                    chunk_length = random.randint(l_scale*self.chunk_length, r_scale*self.chunk_length)
                    chunk_length = chunk_length - chunk_length % 4

                    bs, tmax, idim = xs_pad.shape
                    to_pad = chunk_length * (math.ceil(tmax/chunk_length)) - tmax + 4

                    # ilens += to_pad
                    # Emask = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

                    xs_pad = torch.cat((xs_pad, torch.zeros([bs, to_pad, idim]).to(xs_pad.device)), 1)
                    # tmax_ = tmax + to_pad - 4
                    tmax_ =  chunk_length * (math.ceil(tmax/chunk_length))

                    Cmask, num_chunks = chunk_level_mask(chunk_length=chunk_length, encoder_look_forward=0, padded_length=tmax_, down_sampling=4)

                    Emask_pad = torch.ones([bs, 1, to_pad]).eq(0).to(xs_pad.device) # generate False block and add it to Emask
                    Emask = torch.cat([Emask, Emask_pad], dim=-1)
            else:
                # print('decoding branch')
                # pdb.set_trace()
                if self.simulate_streaming:
                    # forward_chunk_by_chunk
                    raise NotImplementedError
                else:
                    bs, tmax, idim = xs_pad.shape
                    to_pad = self.decoding_chunk_length * (math.ceil(tmax/self.decoding_chunk_length)) - tmax + 4

                    # ilens += to_pad
                    xs_pad = torch.cat((xs_pad, torch.zeros([bs, to_pad, idim]).to(xs_pad.device)), 1)
                    # tmax_ = tmax + to_pad - 4
                    tmax_ =  self.decoding_chunk_length * (math.ceil(tmax/self.decoding_chunk_length))

                    Cmask, num_chunks = chunk_level_mask(chunk_length=self.decoding_chunk_length, encoder_look_forward=0, padded_length=tmax_, down_sampling=4)

                    Emask_pad = torch.ones([bs, 1, to_pad]).eq(0).to(xs_pad.device) # generate False block and add it to Emask
                    Emask = torch.cat([Emask, Emask_pad], dim=-1)
                # pdb.set_trace()
        else:
            Cmask = None
        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            xs_pad, Emask = self.embed(xs_pad, Emask)
        else:
            xs_pad = self.embed(xs_pad)
        #pdb.set_trace()
        xs_pad, Emask, _ = self.causal_encoders(xs_pad, Emask, Cmask)
        if self.decoding:
            if self.non_causal == True:
                xs_pad, masks = self.non_causal_encoders(xs_pad, Emask)
        else:
            random_val = torch.rand(1)
            if self.causal_weight < random_val:
                xs_pad, masks = self.non_causal_encoders(xs_pad, Emask)
        #pdb.set_trace()
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if self.chunk_mode == 'dynamic-u2':
            olens = Emask[:,:,0].sum(1)
        else:
            olens = Emask.squeeze(1).sum(1)
            olens = torch.ones(Emask.squeeze(1).size(), dtype=olens.dtype, device=olens.device).sum(1)
        # print(olens, olens.shape)
        return xs_pad, olens, None

    #     def forward_chunk_by_chunk(
    #     self,
    #     xs: torch.Tensor,
    #     decoding_chunk_size: int,
    #     num_decoding_left_chunks: int = -1,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
