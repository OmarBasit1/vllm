# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, Optional

import torch


class LogprobsLists(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: list[list[int]]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: list[list[float]]
    # [num_reqs]
    sampled_token_ranks: list[int]

    def slice(self, start: int, end: int):
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )

    @staticmethod
    def empty_cpu(num_positions: int,
                  num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.int32,
            device="cpu")
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions,
                                           dtype=torch.int32,
                                           device="cpu")
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


@dataclass
class MoEBlockProfilingResult:
    time_block_start: float
    time_dispatch_end: float
    time_mlp_end: float
    time_combine_end: float
    topk_ids: list[list[int]]  # [num_tokens_in_batch, k]


MoEModelProfilingResult = list[MoEBlockProfilingResult]


def moe_model_profiling_result_to_dict(
        model_result: MoEModelProfilingResult) -> dict:
    """
    Append each key with block layer id.
    """
    ret: dict[str, Any] = {}
    for i, block_result in enumerate(model_result):
        ret |= {f'{k}_{i}': v for k, v in asdict(block_result).items()}
    return ret


@dataclass
class SamplerOutput:

    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Optional[LogprobsTensors]


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass
class ModelRunnerOutput:

    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]]

    # num_reqs x num_spec_tokens
    spec_token_ids: Optional[list[list[int]]]

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: Optional[LogprobsLists]

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]]

    # [req_ids]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None

    moe_model_profiling_result: Optional[MoEModelProfilingResult] = None


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[],
                                              req_id_to_index={},
                                              sampled_token_ids=[],
                                              spec_token_ids=None,
                                              logprobs=None,
                                              prompt_logprobs_dict={},
                                              finished_sending=None,
                                              finished_recving=None)
