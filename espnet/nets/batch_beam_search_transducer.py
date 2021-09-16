"""Parallel beam search module."""

import pdb
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.batch_beam_search import BatchHypothesis
from espnet.nets.e2e_asr_common import end_detect


class BatchBeamSearchTransducer(BatchBeamSearch):
    """Batch beam search implementation."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        print_log: bool = True
    ):
        super().__init__(scorers, weights, beam_size, vocab_size, sos, eos, token_list, pre_beam_ratio, pre_beam_score_key)
        self.print_log = print_log

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, torch.Tensor],
        full_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis.

        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, torch.Tensor]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.

        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        return new_scores

    def merge_states(self, states: Any) -> Any:
        """Merge states for new hypothesis.

        Args:
            states: states of `self.full_scorers`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.

        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        return new_states

    def init_hyp_transducer(self, x: torch.Tensor) -> BatchHypothesis:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0
        return self.batchfy(
            [
                Hypothesis(
                    score=0.0,
                    scores=init_scores,
                    states=init_states,
                    yseq=torch.tensor([self.sos, 0], device=x.device),
                )
            ]
        )

    def greedy_search(self, running_hyps: BatchHypothesis, x: torch.Tensor) -> BatchHypothesis:
        # greedy search
        hyps = self.unbatchfy(running_hyps)
        scores, states = self.score_full(running_hyps, x[0])

        for xi in x :
            ytu = torch.log_softmax(self.full_scorers['rnnt_decoder'].joint_network(xi, scores['rnnt_decoder']), dim=-1)
            top_k = ytu.topk(self.beam_size, dim=-1)
            scores_top = (
                    top_k[0].squeeze(0),
                    top_k[1].squeeze(0),
                )
            for logp, new_id in zip(*scores_top):
                if new_id != self.full_scorers['rnnt_decoder'].blank :
                    scores['rnnt_decoder'] = ytu
                    new_hyp = Hypothesis(
                            score=(hyps[0].score + self.weights['rnnt_decoder'] * float(logp)),
                            yseq=self.append_token(hyps[0].yseq, new_id),
                            scores=self.merge_scores(
                                hyps[0].scores,
                                {k: v[0] for k, v in scores.items()},
                                new_id,
                            ),
                            states=self.merge_states(
                                {
                                    k: self.full_scorers[k].select_state(v, 0)
                                    for k, v in states.items()
                                },
                            ),
                        )
                    del hyps[0]
                    hyps.append(new_hyp)
                    scores, states = self.score_full(self.batchfy(hyps), xi)

        return self.batchfy(hyps)

    # def beam_search(self, running_hyps: BatchHypothesis, x: torch.Tensor) -> BatchHypothesis:
    #     """Search new tokens for running hypotheses and encoded speech x.

    #     Args:
    #         running_hyps (BatchHypothesis): Running hypotheses on beam
    #         x (torch.Tensor): Encoded speech feature (T, D)

    #     Returns:
    #         BatchHypothesis: Best sorted hypotheses

    #     """

    #     init_tensor = x.unsqueeze(0)
    #     blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)
    #     hyps = self.unbatchfy(running_hyps)
    #     # pdb.set_trace()

    #     # beam search
    #     kept_hyps = []
    #     while True:

    #         # pdb.set_trace()
    #         max_hyp = max(hyps, key=lambda x: x.score)
    #         idx = 0
    #         for i in range(len(hyps)):
    #             if max_hyp.score == hyps[i].score:
    #                 idx = i
    #         del hyps[idx]

    #         # hyps.remove(max_hyp)
    #         scores, states = self.score_full(self.batchfy([max_hyp]), x.expand(1, *x.shape))
    #         scores['rnnt_decoder'] = torch.log_softmax(self.full_scorers['rnnt_decoder'].joint_network(x, scores['rnnt_decoder']), dim=-1)
    #         top_k = scores["rnnt_decoder"][:,1:].topk(self.beam_size, dim=-1)
    #         scores_top = (
    #             torch.cat((top_k[0].squeeze(0), scores["rnnt_decoder"][0,0:1])),
    #             torch.cat(((top_k[1]+1).squeeze(0), blank_tensor)),
    #         )

    #         # TODO(karita): do not use list. use batch instead
    #         # see also https://github.com/espnet/espnet/pull/1402#discussion_r354561029
    #         # update hyps

    #         for logp, new_id in zip(*scores_top):

    #             if new_id == self.full_scorers['rnnt_decoder'].blank:
    #                 new_hyp = Hypothesis(
    #                     score=(max_hyp.score + self.weights['rnnt_decoder'] * float(logp)),
    #                     yseq=max_hyp.yseq,
    #                     scores=self.merge_scores(
    #                         max_hyp.scores,
    #                         {k: v[0] for k, v in scores.items()},
    #                         new_id,
    #                     ),
    #                     states=max_hyp.states,
    #                 )
    #                 kept_hyps.append(new_hyp)
    #             else:
    #                 new_score = max_hyp.score + self.weights['rnnt_decoder'] * float(logp)
    #                 if self.weights.get('lm') is not None :
    #                     new_score += self.weights['lm'] * scores['lm'][0][new_id]
    #                 new_hyp = Hypothesis(
    #                     score=new_score,
    #                     # score=(max_hyp.score + self.weights['rnnt_decoder'] * float(logp)),
    #                     yseq=self.append_token(max_hyp.yseq, new_id),
    #                     scores=self.merge_scores(
    #                         max_hyp.scores,
    #                         {k: v[0] for k, v in scores.items()},
    #                         new_id,
    #                     ),
    #                     states=self.merge_states(
    #                         {
    #                             k: self.full_scorers[k].select_state(v, 0)
    #                             for k, v in states.items()
    #                         },
    #                     ),
    #                 )
    #                 hyps.append(new_hyp)
    #         hyps_max = float(max(hyps, key=lambda x: x.score).score)
    #         best_most_prob = sorted(
    #             [hyp for hyp in kept_hyps if float(hyp.score) > hyps_max],
    #             key=lambda x: x.score,
    #         )
    #         if len(best_most_prob) >= self.beam_size:
    #             return self.batchfy(best_most_prob)

    def beam_search(self, running_hyps: BatchHypothesis, x: torch.Tensor) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """
        kept_hyps = self.unbatchfy(running_hyps)
        for xi in x:
            # pdb.set_trace()
            hyps = kept_hyps
            kept_hyps = []
            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                # remove max_hyp from hyps
                idx = 0
                for i in range(len(hyps)):
                    if max_hyp.score == hyps[i].score:
                        idx = i
                del hyps[idx]

                scores, states = self.score_full(self.batchfy([max_hyp]), xi)
                scores['rnnt_decoder'] = torch.log_softmax(self.full_scorers['rnnt_decoder'].joint_network(xi, scores['rnnt_decoder']), dim=-1)
                top_k = scores["rnnt_decoder"][0,1:].topk(self.beam_size, dim=-1)
                new_hyp = Hypothesis(
                    score=(max_hyp.score + self.weights['rnnt_decoder'] * float(scores["rnnt_decoder"][0,0:1])),
                    yseq=max_hyp.yseq,
                    scores=self.merge_scores(
                        max_hyp.scores,
                        {k: v[0] for k, v in scores.items()},
                        self.full_scorers['rnnt_decoder'].blank,
                    ),
                    states=max_hyp.states,
                )
                kept_hyps.append(new_hyp)
                for logp, new_id in zip(*top_k):
                    score = max_hyp.score + self.weights['rnnt_decoder'] * float(logp)
                    if self.weights.get('lm') is not None :
                        score += self.weights['lm'] * scores['lm'][0][new_id+1]
                    new_hyp = Hypothesis(
                        score=score,
                        # score=(max_hyp.score + self.weights['rnnt_decoder'] * float(logp)),
                        yseq=self.append_token(max_hyp.yseq, new_id+1),
                        scores=self.merge_scores(
                            max_hyp.scores,
                            {k: v[0] for k, v in scores.items()},
                            new_id,
                        ),
                        states=self.merge_states(
                            {
                                k: self.full_scorers[k].select_state(v, 0)
                                for k, v in states.items()
                            },
                        ),
                    )
                    hyps.append(new_hyp)
                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if float(hyp.score) > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= self.beam_size:
                    kept_hyps = kept_most_prob
                    break
        return self.batchfy(kept_hyps)

    def beam_search_improve(self, running_hyps: BatchHypothesis, x: torch.Tensor) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.
        implement by https://arxiv.org/abs/1911.01629

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """
        state_beam = 2.3
        expand_beam = 2.3
        kept_hyps = self.unbatchfy(running_hyps)
        for xi in x:
            # pdb.set_trace()
            hyps = kept_hyps
            kept_hyps = []
            while True:
                if len(kept_hyps) >= self.beam_size:
                    break
                a_best_hyp = max(hyps, key=lambda x: x.score / len(x.yseq))
                # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                if len(kept_hyps) > 0:
                    b_best_hyp = max(kept_hyps, key=lambda x: x.score / len(x.yseq))
                    a_best_prob = a_best_hyp.score
                    b_best_prob = b_best_hyp.score
                    if b_best_prob >= state_beam + a_best_prob:
                        break
                # remove max_hyp from hyps
                idx = 0
                for i in range(len(hyps)):
                    if a_best_hyp.score == hyps[i].score:
                        idx = i
                del hyps[idx]

                scores, states = self.score_full(self.batchfy([a_best_hyp]), xi)
                scores['rnnt_decoder'] = torch.log_softmax(self.full_scorers['rnnt_decoder'].joint_network(xi, scores['rnnt_decoder']), dim=-1)
                logp_targets, positions = scores["rnnt_decoder"].view(-1).topk(self.beam_size, dim=-1)
                best_logp = (
                    logp_targets[0]
                    if positions[0] != self.full_scorers['rnnt_decoder'].blank
                    else logp_targets[1]
                )
                for j in range(logp_targets.size(0)):
                    score = a_best_hyp.score + self.weights['rnnt_decoder'] * float(logp_targets[j])
                    if positions[j] == self.full_scorers['rnnt_decoder'].blank:
                        topk_hyp = Hypothesis(
                            score=score,
                            yseq=a_best_hyp.yseq,
                            scores=self.merge_scores(
                                a_best_hyp.scores,
                                {k: v[0] for k, v in scores.items()},
                                self.full_scorers['rnnt_decoder'].blank,
                            ),
                            states=a_best_hyp.states,
                        )
                        kept_hyps.append(topk_hyp)
                        continue
                    if logp_targets[j] >= best_logp - expand_beam:
                        if self.weights.get('lm') is not None :
                            score += self.weights['lm'] * scores['lm'][0][positions[j]]
                        topk_hyp = Hypothesis(
                            score=score,
                            yseq=self.append_token(a_best_hyp.yseq, positions[j]),
                            scores=self.merge_scores(
                                a_best_hyp.scores,
                                {k: v[0] for k, v in scores.items()},
                                positions[j],
                            ),
                            states=self.merge_states(
                                {
                                    k: self.full_scorers[k].select_state(v, 0)
                                    for k, v in states.items()
                                },
                            ),
                        )
                        hyps.append(topk_hyp)
        nbest_hyps  = sorted(
            [hyp for hyp in kept_hyps],
            key=lambda x: x.score / len(x.yseq),
            reverse=True,
        )[:self.beam_size]
        return self.batchfy(nbest_hyps)

    def post_process(
        self,
        running_hyps: BatchHypothesis,
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.

        Args:
            running_hyps (BatchHypothesis): The running hypotheses in beam search.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_batch = running_hyps.yseq.shape[0]
        logging.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[x]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )
        # add eos in the final loop to avoid that there are no ended hyps
        yseq_eos = torch.cat(
            (
                running_hyps.yseq,
                torch.full(
                    (n_batch, 1),
                    self.eos,
                    device=running_hyps.yseq.device,
                    dtype=torch.int64,
                ),
            ),
            1,
        )
        running_hyps.yseq.resize_as_(yseq_eos)
        running_hyps.yseq[:] = yseq_eos
        running_hyps.length[:] = yseq_eos.shape[1]
        return running_hyps

    def forward(
        self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        getattr(self, "print_log", setattr(self, "print_log", True))
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        if self.print_log:
            logging.info("decoder input length: " + str(x.shape[0]))
            logging.info("max output length: " + str(maxlen))
            logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        kept_hyps = self.init_hyp_transducer(x)
        # ended_hyps = []
        if self.beam_size <= 1:
            kept_hyps = self.greedy_search(kept_hyps, x)
        else:
            kept_hyps = self.beam_search_improve(kept_hyps, x)
            # kept_hyps = self.beam_search(kept_hyps, x)
        kept_hyps = self.post_process(kept_hyps)
        # nbest_hyps = sorted(kept_hyps, key=lambda x: x.score, reverse=True)
        nbest_hyps = sorted(self.unbatchfy(kept_hyps), key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        if self.print_log:
            best = nbest_hyps[0]
            for k, v in best.scores.items():
                logging.info(
                    f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
                )
            logging.info(f"total log probability: {best.score:.2f}")
            logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
            logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
            if self.token_list is not None:
                logging.info(
                    "best hypo: "
                    + "".join([self.token_list[x] for x in best.yseq[2:-1] if x != 1 and x != self.eos])
                    + "\n"
                )
        return nbest_hyps
