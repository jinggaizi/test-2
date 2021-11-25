# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import Optional, List
from typing import Tuple

import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.streaming_u2.encoder_layer import EncoderLayer
from espnet2.asr.streaming_u2.attention import MultiHeadedAttention
from espnet2.asr.streaming_u2.attention import RelPositionMultiHeadedAttention
from espnet2.asr.streaming_u2.convolution import ConvolutionModule
from espnet2.asr.streaming_u2.subsampling import Conv2dSubsampling4
from espnet2.asr.streaming_u2.embedding import PositionalEncoding
from espnet2.asr.streaming_u2.embedding import RelPositionalEncoding
from espnet2.asr.streaming_u2.mask import add_optional_chunk_mask
import pdb



class ConformerStreamingCascadedU2Encoder(AbsEncoder):
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
        padding_idx: int = -1,
        use_causal_cnn: bool = True,
        cnn_module_norm: str = "batch_norm",
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        use_non_causal_layer_decoding: bool = False,
        causal_weight: float = 0.3,
        decoding: bool = False,
        decoding_chunk_length: int = 10, 
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        intermediate_causal: bool = False,
    
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn" 
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.embed = Conv2dSubsampling4(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
        )

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
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_causal_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, use_causal_cnn)
        convolution_non_causal_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm)

        self.causal_encoders = repeat(
            num_causal_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_causal_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.non_causal_encoders = repeat(
            num_non_causal_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_non_causal_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        # streaming related
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.static_chunk_size = static_chunk_size
        self.use_non_causal_layer_decoding = use_non_causal_layer_decoding
        self.causal_weight = causal_weight
        self.decoding = decoding
        self.simulate_streaming = simulate_streaming
        self.decoding_chunk_length = decoding_chunk_length
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.intermediate_causal = intermediate_causal

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
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        # Emask_ = Emask

        #pdb.set_trace()
        if self.decoding:
            if self.simulate_streaming:
                xs_pad, pos_emb, masks = self.forward_chunk_by_chunk(xs_pad, self.decoding_chunk_length, self.num_decoding_left_chunks)
                olens = masks.squeeze(1).sum(1)
                olens = torch.ones(masks.squeeze(1).size(), dtype=olens.dtype, device=olens.device).sum(1)
                if self.use_non_causal_layer_decoding == True:
                    for layer in self.non_causal_encoders:
                        xs_pad, masks, _ = layer(xs_pad, masks, pos_emb)
                    if self.normalize_before:
                        xs_pad = self.after_norm(xs_pad)
                    return xs_pad, olens, None
                else:
                    return xs_pad, olens, None
            else:
                xs_pad, pos_emb, masks = self.embed(xs_pad, masks)
                chunk_masks = add_optional_chunk_mask(xs_pad, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              self.decoding_chunk_length,
                                              self.static_chunk_size,
                                              self.num_decoding_left_chunks)
                for layer in self.causal_encoders:
                    xs_pad, chunk_masks, _ = layer(xs_pad, chunk_masks, pos_emb, masks)
                if self.use_non_causal_layer_decoding:
                    for layer in self.non_causal_encoders:
                        xs_pad, masks, _ = layer(xs_pad, masks, pos_emb)
                if self.normalize_before:
                    xs_pad = self.after_norm(xs_pad)
                olens = masks.squeeze(1).sum(1)
                return xs_pad, olens, None

        xs_pad, pos_emb, masks = self.embed(xs_pad, masks)
        chunk_masks = add_optional_chunk_mask(xs_pad, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              0,
                                              self.static_chunk_size,
                                              -1)
       
        #pdb.set_trace()
        for layer in self.causal_encoders:
            xs_pad, chunk_masks, _,   = layer(xs_pad, chunk_masks, pos_emb, masks)

        if self.intermediate_causal:
            if self.normalize_before:
                xs_pad_causal = self.after_norm(xs_pad)
            for layer in self.non_causal_encoders:
                xs_pad, masks, _ = layer(xs_pad, masks, pos_emb)
            if self.normalize_before:
               xs_pad = self.after_norm(xs_pad) 
            olens = masks.squeeze(1).sum(1)
            return xs_pad, olens, xs_pad_causal       
        else:
            random_val = torch.rand(1)
            if self.causal_weight < random_val:
                for layer in self.non_causal_encoders:
                    xs_pad, masks, _ = layer(xs_pad, masks, pos_emb)
        
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)

            olens = masks.squeeze(1).sum(1)
            
            return xs_pad, olens, None

    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        subsampling_cache: Optional[torch.Tensor] = None,
        elayers_output_cache: Optional[List[torch.Tensor]] = None,
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor],
               List[torch.Tensor]]:
        """ Forward just one chunk
        Args:
            xs (torch.Tensor): chunk input
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            subsampling_cache (Optional[torch.Tensor]): subsampling cache
            elayers_output_cache (Optional[List[torch.Tensor]]):
                transformer/conformer encoder layers output cache
            conformer_cnn_cache (Optional[List[torch.Tensor]]): conformer
                cnn cache
        Returns:
            torch.Tensor: output of current input xs
            torch.Tensor: subsampling cache required for next chunk computation
            List[torch.Tensor]: encoder layers output cache required for next
                chunk computation
            List[torch.Tensor]: conformer cnn cache
        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        xs, pos, _ = self.embed(xs, tmp_masks, offset)
        if subsampling_cache is not None:
            cache_size = subsampling_cache.size(1)
            xs = torch.cat((subsampling_cache, xs), dim=1)
        else:
            cache_size = 0
        pos_emb = self.embed.position_encoding(offset - cache_size, xs.size(1))
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = xs.size(1)
        else:
            next_cache_start = max(xs.size(1) - required_cache_size, 0)
        r_subsampling_cache = xs[:, next_cache_start:, :]
        # Real mask for transformer/conformer layers
        masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        r_elayers_output_cache = []
        r_conformer_cnn_cache = []
        import pdb
        #pdb.set_trace()
        for i, layer in enumerate(self.causal_encoders):
            if elayers_output_cache is None:
                attn_cache = None
            else:
                attn_cache = elayers_output_cache[i]
            if conformer_cnn_cache is None:
                cnn_cache = None
            else:
                cnn_cache = conformer_cnn_cache[i]
            xs, _, new_cnn_cache = layer(xs,
                                         masks,
                                         pos_emb,
                                         cache=attn_cache,
                                         cnn_cache=cnn_cache)
            r_elayers_output_cache.append(xs[:, next_cache_start:, :])
            r_conformer_cnn_cache.append(new_cnn_cache)
        if self.use_non_causal_layer_decoding == True:
            pass
        else:
            if self.normalize_before:
                xs = self.after_norm(xs)
        import pdb
        #pdb.set_trace()

        return (xs[:, cache_size:, :], r_subsampling_cache,
                r_elayers_output_cache, r_conformer_cnn_cache, pos_emb[:, cache_size:, :])
    
    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_length: int,
        num_decoding_left_chunks: int = -1,
        #num_decoding_right_chunks: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion
        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling
        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_length > 0
        # The model is trained by static or dynamic chunk
        #assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_length
        decoding_window = (decoding_chunk_length - 1) * subsampling + context
        bs, num_frames, idim = xs.shape
        subsampling_cache: Optional[torch.Tensor] = None
        elayers_output_cache: Optional[List[torch.Tensor]] = None
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        outputs = []
        outputs_pos = []
        offset = 0
        required_cache_size = decoding_chunk_length * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]

            (y, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache, pos_emb) = self.forward_chunk(chunk_xs, offset,
                                                       required_cache_size,
                                                       subsampling_cache,
                                                       elayers_output_cache,
                                                       conformer_cnn_cache)
            outputs.append(y)
            outputs_pos.append(pos_emb)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        pos = torch.cat(outputs_pos, 1)
        masks = torch.ones(1, ys.size(1), device=ys.device, dtype=torch.bool)
        masks = masks.unsqueeze(1)
        return ys, pos, masks

