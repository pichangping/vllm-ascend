#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
                if self.vllm_config.model_config.use_mla:
                    long_seq_metadata.kv_with_q_head_nomask_idx_tensor = split_q_head_nomask_idx_tensor_list
                    long_seq_metadata.kv_with_q_tail_nomask_idx_tensor = split_q_tail_nomask_idx_tensor_list
                    long_seq_metadata.head_attn_nomask_seqlens = head_attn_nomask_seqlens_list
                    long_seq_metadata.tail_attn_nomask_seqlens = tail_attn_nomask_seqlens_list

    def _list_to_tensor(self, lst, device, dtype=torch.int32):
        tensor_npu = torch.zeros(len(lst), dtype=dtype, device=device)
        tensor_npu.copy_(torch.tensor(lst, dtype=dtype),
                            non_blocking=True)
        return tensor_npu

    mock_runner._get_cp_local_seq_lens.side_effect = NPUModelRunner._get_cp_local_seq_lens.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._list_to_tensor.side_effect = NPUModelRunner._list_to_tensor.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._split_nomask_idx_tensor_list.side_effect = NPUModelRunner._split_nomask_idx_tensor_list.__get__(
        mock_runner, NPUModelRunner)
    mock_runner._split_multi_batch_kv_idx.side_effect = NPUModelRunner._split_multi_batch_kv_idx.__get__(
        mock_runner, NPUModelRunner)

    result = NPUModelRunner._generate_pcp_metadata(mock_runner, total_tokens)

        if len(kv_nomask_idx[0]) == 0:
            return attn_output, attn_lse
        for kv_nomask_idx_split, attn_nomask_seqlens_split in zip(
                kv_nomask_idx, attn_nomask_seqlens):
            k_nope_nomask = torch.index_select(k_nope, 0, kv_nomask_idx_split)
            value_nomask = torch.index_select(value, 0, kv_nomask_idx_split)
            k_pe_nomask = torch.index_select(k_pe, 0, kv_nomask_idx_split)
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope_nomask,
                k_rope=k_pe_nomask,
                value=value_nomask,
                mask=mask,
                seqlen=attn_nomask_seqlens_split,
                head_num=self.num_heads,
                kv_head_num=self.num_heads,
                pre_out=attn_output,
                prev_lse=attn_lse,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="no_mask",
                input_layout="type_bsnd",
                calc_type="calc_type_default",
                output=attn_output,
                softmax_lse=attn_lse)
        return attn_output, attn_lse

        kv_mask_idx :list[torch.Tensor],
        kv_nomask_idx :list[torch.Tensor],

@pytest.mark.parametrize(
    "pcp_rank, split_with_q_head_nomask_idx_reqs, split_kv_with_q_tail_nomask_idx_reqs,"
    "head_attn_nomask_seqlens, chunk_seqlens,"
    "target_split_q_head, target_split_q_tail, target_head_seqlens, target_tail_seqlens",
    [
        # case1: pcp_rank=0
        (
            0,
            [[10, 20, 30]],
            [[40, 50, 60]],
            torch.tensor([[64], [0]], dtype=torch.int32),
            [64],
            [torch.tensor([1, 2, 3], dtype=torch.int32)],
            [torch.tensor([40, 50, 60], dtype=torch.int32)],
            [torch.tensor([[64], [0]], dtype=torch.int32)],
            [torch.tensor([[64], [3]], dtype=torch.int32)]
        ),
        # case2: pcp_rank=1
        (
            1,
            [[1, 2], [3, 4, 5]],
            [[6, 7], [8, 9, 10]],
            torch.tensor([[128, 128], [128, 128]], dtype=torch.int32),
            [128, 128],
            [torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)],
            [torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32)],
            [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)],
            [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)]
        ),
        # case3: pcp_rank=2
        (
            2,
            [[11, 12, 13, 14], [15, 16]],
            [[17, 18, 19], [20, 21, 22, 23]],
            torch.tensor([[256, 256], [512, 512]], dtype=torch.int32),
            [256, 256],
            [torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int32)],
            [torch.tensor([17, 18, 19, 20, 21, 22, 23], dtype=torch.int32)],
            [torch.tensor([[256, 256], [4, 2]], dtype=torch.int32)],
            [torch.tensor([[256, 256], [3, 4]], dtype=torch.int32)]
        ),
        # case4: empty input
        (
            0,
            [],
            [],
            torch.tensor([], dtype=torch.int32).reshape(2, 0),
            [],
            [],
            [],
            [],
            [],
        ),
        # case5: single element input
        (
            0,
            [[10]],
            [[40]],
            torch.tensor([[64], [0]], dtype=torch.int32),
            [64],
            [torch.tensor([1, 2, 3], dtype=torch.int32)],  
            [torch.tensor([40], dtype=torch.int32)],
            [torch.tensor([[64], [0]], dtype=torch.int32)],
            [torch.tensor([[64], [1]], dtype=torch.int32)],
        ),
        # case6: pcp_rank=3
        (
            3,
            [[1, 2], [3, 4, 5]],
            [[6, 7], [8, 9, 10]],
            torch.tensor([[128, 128], [128, 128]], dtype=torch.int32),
            [128, 128],
            [torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)],
            [torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32)],
            [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)], 
            [torch.tensor([[128, 128], [2, 3]], dtype=torch.int32)],
        ),
    ]
)
def test_split_nomask_idx_tensor_list(
    pcp_rank,
    split_with_q_head_nomask_idx_reqs,
    split_kv_with_q_tail_nomask_idx_reqs,
    head_attn_nomask_seqlens,
    chunk_seqlens,
    target_split_q_head,
    target_split_q_tail,
    target_head_seqlens,
    target_tail_seqlens
):
    # Mock input data
    mock_runner = MagicMock(spec=NPUModelRunner)
    mock_runner.device = "cpu"
    mock_runner.pcp_rank = 0
    mock_runner.kv_idx_names = {
        "kv_with_q_head_nomask_idx_tensor": torch.tensor([1, 2, 3], dtype=torch.int32)
    }

    mock_runner.pcp_rank = pcp_rank

    # Mock output
    mock_runner._split_multi_batch_kv_idx.side_effect = NPUModelRunner._split_multi_batch_kv_idx.__get__(mock_runner, NPUModelRunner)
    mock_runner._list_to_tensor.side_effect = NPUModelRunner._list_to_tensor.__get__(mock_runner, NPUModelRunner)

    # Call the method under test
    result = NPUModelRunner._split_nomask_idx_tensor_list(
        mock_runner,
        split_with_q_head_nomask_idx_reqs=split_with_q_head_nomask_idx_reqs,
        split_kv_with_q_tail_nomask_idx_reqs=split_kv_with_q_tail_nomask_idx_reqs,
        head_attn_nomask_seqlens=head_attn_nomask_seqlens,
        chunk_seqlens=chunk_seqlens
    )
    split_q_head, split_q_tail, head_seqlens, tail_seqlens = result

    # Assert the method call
    assert len(split_q_head) == len(target_split_q_head)
    for res, target in zip(split_q_head, target_split_q_head):
        assert torch.equal(res, target)


    assert len(split_q_tail) == len(target_split_q_tail)
    for res, target in zip(split_q_tail, target_split_q_tail):
        assert torch.equal(res, target)


    assert len(head_seqlens) == len(target_head_seqlens)
    for res, target in zip(head_seqlens, target_head_seqlens):
        if isinstance(target, torch.Tensor):
            assert torch.equal(res, target)
        else:
            assert res == target


    assert len(tail_seqlens) == len(target_tail_seqlens)
    for res, target in zip(tail_seqlens, target_tail_seqlens):
        if isinstance(target, torch.Tensor):
            assert torch.equal(res, target)
        else:
            assert res == target


@pytest.mark.parametrize(
    "kv_nomask_idx_multi_batch, split_size, expected_merged_idx, expected_merged_len",
    [
        # case1: multiple batches + split size greater than batch length
        (
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7]
            ],
            2,
            # expected  merged_split_kv_idx_3d
            [
                [0, 1, 5, 6],
                [2, 3, 7],
                [4]
            ],
            # expected merged_split_kv_len_2d
            [
                [2, 2],
                [2, 1],
                [1, 0]
            ],
        ),
        # case2: single batch + split size greater than batch length
        (
            [
                [0, 1, 2]
            ],
            5,
            [
                [0, 1, 2]
            ],
            [
                [3]
            ],
        ),
        # case3: split size equals maximum batch length
        (
            [
                [0, 1, 2, 3],
                [5, 6]
            ],
            4,
            [
                [0, 1, 2, 3, 5, 6]
            ],
            [
                [4, 2]
            ],
        ),
        # case4: Split size is 1 (minimum granularity split)
        (
            [
                [0, 1],
                [2]
            ],
            1,
            [
                [0, 2],
                [1]
            ],
            [
                [1, 1],
                [1, 0]
            ],
        ),
        # case6: the batch contains an empty list
        (
            [
                [],
                [0, 1],
                [2]
            ],
            1,
            [
                [0, 2],
                [1]
            ],
            [
                [0, 1, 1],
                [0, 1, 0]
            ],
        ),
    ]
)
def test_split_multi_batch_kv_idx(
    kv_nomask_idx_multi_batch,
    split_size,
    expected_merged_idx,
    expected_merged_len,
):
    # Mock input data
    model_runner = MagicMock(spec=NPUModelRunner)

    # Call the method under test
    result = NPUModelRunner._split_multi_batch_kv_idx(
        self=model_runner,
        kv_nomask_idx_multi_batch=kv_nomask_idx_multi_batch,
        split_size=split_size
    )

    merged_split_kv_idx_3d, merged_split_kv_len_2d = result

    # Assert the method call
    assert len(merged_split_kv_idx_3d) == len(expected_merged_idx)

    for t, (actual_seg, expected_seg) in enumerate(zip(merged_split_kv_idx_3d, expected_merged_idx)):
        assert actual_seg == expected_seg

    assert len(merged_split_kv_len_2d) == len(expected_merged_len)

    for t, (actual_len, expected_len) in enumerate(zip(merged_split_kv_len_2d, expected_merged_len)):
        assert actual_len == expected_len

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, is_enable_nz,
                               vllm_version_is)

if vllm_version_is("0.11.0"):
    from vllm.config import CompilationLevel
else:
    from vllm.config import CompilationMode


class AscendW8A8DynamicLinearMethod:
    """Linear method for Ascend W8A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        return params_dict

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        return {}

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        config = getattr(layer, "_ascend_quant_config", {})
        if not isinstance(x, tuple):
            output_dtype = config.get("output_dtype", x.dtype)
            quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        else:
            assert "output_dtype" in config.keys(), (
                f"DynamicLinearMethod needs explicitly specified `output_dtype`"
                f"for pre-quantized input, got config [{config}]")
            output_dtype = config["output_dtype"]
            quantized_x, dynamic_scale = x
        pertoken_scale = (dynamic_scale
                          if config.get("pertoken_scale", True) else None)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=output_dtype,
        )
        return ((output, dynamic_scale)
                if config.get("return_scale", False) else output)

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # cast quantized weight tensors in NZ format for higher inference speed
        if is_enable_nz():
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()


class AscendW8A8DynamicFusedMoEMethod:
    """FusedMoe method for Ascend W8A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        if vllm_version_is("0.11.0"):
            self.use_aclgraph = (
                vllm_config.compilation_config.level
                == CompilationLevel.PIECEWISE
                and not vllm_config.model_config.enforce_eager
                and not ascend_config.torchair_graph_config.enabled)
        else:
            self.use_aclgraph = (
                vllm_config.compilation_config.mode
                == CompilationMode.VLLM_COMPILE
                and not vllm_config.model_config.enforce_eager
                and not ascend_config.torchair_graph_config.enabled)

        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""

    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 *
                                               intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.int8)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)
        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    1,
                                                    dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts,
                                                     hidden_sizes,
                                                     1,
                                                     dtype=params_dtype)
        return param_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts - global_redundant_expert_num, "Number of global experts mismatch (excluding redundancy)"

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w1_scale=layer.w13_weight_scale_fp32,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int8_w8a8=True,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None))

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.w13_weight.data = layer.w13_weight.data.transpose(
                1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(
                1, 2).contiguous()
        if is_enable_nz():
            torch_npu.npu_format_cast_(layer.w13_weight, ACL_FORMAT_FRACTAL_NZ)
            torch_npu.npu_format_cast_(layer.w2_weight, ACL_FORMAT_FRACTAL_NZ)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
            layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(
            torch.float32)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(
            layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(
            layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(
            layer.w2_weight_offset.data.shape[0], -1)
