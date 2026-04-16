# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.moe_profiler import MoEProfiler


def test_filter_non_local_experts_keeps_global_local_ids() -> None:
    expert_ids = torch.tensor(
        [
            [0, 2, 7],
            [1, -1, 3],
        ],
        dtype=torch.int32,
    )
    local_expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)

    mapped_ids = MoEProfiler._filter_non_local_experts(
        expert_ids,
        local_expert_map,
    )

    expected_ids = torch.tensor(
        [
            [-1, 2, -1],
            [1, -1, -1],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(mapped_ids, expected_ids)


def test_materialize_iteration_record_keeps_only_local_experts() -> None:
    record = {
        "iteration_id": 7,
        "token_count": 3,
        "layer_ids": [4],
        "request_token_counts": [2, 1],
        "expert_ids": torch.tensor(
            [
                [[0, -1, 2]],
                [[-1, 3, -1]],
                [[4, 5, -1]],
            ],
            dtype=torch.int32,
        ),
    }

    materialized = MoEProfiler._materialize_iteration_record(record)

    assert materialized["iteration_no"] == 7
    assert materialized["layers"] == [
        {
            "layer_no": 4,
            "request_expert_ids": [
                [[0, 2], [3]],
                [[4, 5]],
            ],
        }
    ]
