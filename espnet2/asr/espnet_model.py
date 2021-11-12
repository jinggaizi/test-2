from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import argparse
import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.e2e_asr_common import ErrorCalculatorTransducer
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.asr.rnnt_decoder.abs_rnnt_decoder import AbsRNNTDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask_limit

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        decoder,
        decoder_causal,
        ctc,
        rnnt_decoder,
        ctc_weight: float = 0.5,
        rnnt_weight: float = 0.0,
        ctc_causal_weight: float = 0.0,
        att_causal_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        blank_id: int = 0,
        lamb: float = 0.0,
        target_attn_mask_left: int = -1,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        # assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.blank_id = blank_id
        self.ctc_weight = ctc_weight
        self.ctc_causal_weight = ctc_causal_weight
        self.att_causal_weight = att_causal_weight
        self.rnnt_weight = rnnt_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_causal = decoder_causal
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        if rnnt_weight == 0.0:
            self.rnnt_decoder = None
            self.criterion_trans = None
        else:
            self.rnnt_decoder = rnnt_decoder
            self.criterion_trans = TransLoss("warp-rnnt", lamb,  blank_id)
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        # self.criterion_trans = TransLoss("warp-rnnt", lamb,  blank_id)
        self.target_attn_mask_left = target_attn_mask_left
        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
            self.error_calculator_trans = ErrorCalculatorTransducer(
                decoder=self.rnnt_decoder, 
                token_list=self.token_list,
                sym_space=sym_space,
                sym_blank=sym_blank,
                sym_sos=self.sos,
                sym_eos=self.eos,
                report_cer=report_cer,
                report_wer=report_wer
            )
        else:
            self.error_calculator = None
            self.error_calculator_trans = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        if self.decoder_causal:
            encoder_out, encoder_out_lens, encoder_out_causal = self.encode(speech, speech_lengths)
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # # 2a. Attention-decoder branch
        # if self.ctc_weight == 1.0:
        #     loss_att, acc_att, cer_att, wer_att = None, None, None, None
        # else:
        #     if self.rnnt_decoder is not None:
        #         # loss_rnnt, cer_rnnt = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)
        #         loss_att, acc_att, cer_att, wer_att = None, None, None, None
        #     else:
        #         loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
        #             encoder_out, encoder_out_lens, text, text_lengths
        #         )

        # # 2b. CTC branch
        # if self.ctc_weight == 0.0:
        #     loss_ctc, cer_ctc = None, None
        #     if self.rnnt_decoder is not None:
        #         loss_rnnt, cer_rnnt = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)
        #     else:
        #         loss_rnnt, cer_rnnt = None, None
        # else:
        #     # 2c. RNN-T branch
        #     if self.rnnt_decoder is not None:
        #         loss_rnnt, cer_rnnt = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)
        #         loss_ctc, cer_ctc = self._calc_ctc_loss(
        #             encoder_out, encoder_out_lens, text, text_lengths
        #         )
        #     else:
        #         loss_ctc, cer_ctc = self._calc_ctc_loss(
        #             encoder_out, encoder_out_lens, text, text_lengths
        #         )
        #         loss_rnnt, cer_rnnt = None, None

        # if self.ctc_weight == 0.0:
        #     if self.rnnt_decoder is not None:
        #         loss = loss_rnnt
        #     else:
        #         loss = loss_att
        # elif self.ctc_weight == 1.0:
        #     loss = loss_ctc
        # else:
        #     if self.rnnt_decoder is not None:
        #         loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_rnnt
        #     else:
        #         loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        # 2a. Attention-decoder branch
        if (self.ctc_weight + self.rnnt_weight) == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
        # 2c. RNN-T branch
        if self.rnnt_weight == 0.0:
            loss_rnnt, cer_rnnt = None, None
        else:
            loss_rnnt, cer_rnnt = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)
        # causal output loss
        if not self.decoder_causal:
            loss_ctc_causal, cer_ctc_causal = None, None
            loss_att_causal, acc_att_causal, cer_att_causal, wer_att_causal = None, None, None, None
        else:
            loss_ctc_causal, cer_ctc_causal = self._calc_ctc_loss(
                encoder_out_causal, encoder_out_lens, text, text_lengths
            )
            loss_att_causal, acc_att_causal, cer_att_causal, wer_att_causal = self._calc_att_loss(
                encoder_out_causal, encoder_out_lens, text, text_lengths, True
            )
        
        if loss_att is not None:
            loss = (1 - self.ctc_weight - self.rnnt_weight - self.ctc_causal_weight - self.att_causal_weight) * loss_att
        if loss_att_causal is not None:
            loss = loss + self.att_causal_weight * loss_att_causal
        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc
        if loss_ctc_causal is not None:
            loss = loss + self.ctc_causal_weight * loss_ctc_causal
        if loss_rnnt is not None:
            loss = loss + self.rnnt_weight * loss_rnnt
        
        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_att_causal=loss_att_causal.detach() if loss_att_causal is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_ctc_causal=loss_ctc_causal.detach() if loss_ctc_causal is not None else None,
            loss_rnnt=loss_rnnt.detach() if loss_rnnt is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
            cer_rnnt=cer_rnnt,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)
            
        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.decoder_causal:
            assert(self.encoder.intermediate_causal), "decoder_causal use cascaded model and must set intermediate_causal true"
            encoder_out, encoder_out_lens, encoder_out_causal = self.encoder(feats, feats_lengths)
            assert encoder_out.size(0) == speech.size(0), (
                encoder_out.size(),
                speech.size(0),
            )
            return encoder_out, encoder_out_lens, encoder_out_causal
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        '''
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )
        '''

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        causal: bool = False,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        if causal and self.decoder_causal:
            decoder_out, _ = self.decoder_causal(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )
        else:
            decoder_out, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # raise NotImplementedError
        hs_mask = (~make_pad_mask(encoder_out_lens)[:, None, :])
        ys_in_pad, _, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hs_mask)

        # 1. Forward decoder
        if self.target_attn_mask_left == -1: # not use mask
            ys_mask = target_mask(ys_in_pad, self.blank_id)
        else:
            ys_mask = target_mask_limit(ys_in_pad, self.blank_id, self.target_attn_mask_left)
        
        
        decoder_out, _ = self.rnnt_decoder(
            encoder_out, ys_in_pad, ys_mask
        )

        # Calc RNNT loss
        loss_rnnt = self.criterion_trans(decoder_out, target, pred_len, target_len)

        # Calc CER
        if self.training or self.error_calculator_trans is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator_trans(encoder_out, ys_pad)

        return loss_rnnt, cer
