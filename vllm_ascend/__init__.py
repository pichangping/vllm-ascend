#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_model():
    # fix pytorch schema check error, remove this line after pytorch
    # is upgraded to 2.7.0
    import vllm.envs as envs

    import vllm_ascend.patch.worker.patch_common.patch_utils  # noqa: F401

    from .models import register_model

    import vllm_ascend.envs as envs_ascend  # isort: skip  # noqa: F401
    if envs.VLLM_USE_V1 and \
        envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM != 0:
        import vllm_ascend.patch.platform.patch_0_9_1.patch_decorator  # isort: skip  # noqa: F401
    register_model()


def enable_sfa_quant(vllm_config) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if hf_text_config is None:
        return False
    return hasattr(hf_text_config, "index_topk") and not hasattr(hf_text_config, "compress_ratios")

def enable_sfa_quant(vllm_config) -> bool:
    return (
        hasattr(vllm_config.model_config, "hf_text_config")
        and hasattr(vllm_config.model_config.hf_text_config, "index_topk")
        and not hasattr(vllm_config.model_config.hf_text_config, "compress_ratios")
    )


from vllm_ascend.ascend_config import get_ascend_config

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, get_sfa_qsfa_packed_head_dim

from vllm_ascend.quantization.utils import enable_fa_quant, enable_sfa_quant

def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Build Ascend-specific KV cache specs for v2 worker patching."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

    kv_cache_spec: dict[str, KVCacheSpec] = {}
    layer_type = AttentionLayerBase
    attn_layers = get_layers_from_vllm_config(vllm_config, layer_type)

    if get_ascend_device_type() == AscendDeviceType.A5:
        c8_k_cache_dtype = torch.float8_e4m3fn
        c8_k_scale_cache_dtype = torch.float32
    else:
        c8_k_cache_dtype = torch.int8
        c8_k_scale_cache_dtype = torch.float16

    for layer_name, attn_module in attn_layers.items():
        if getattr(attn_module, "kv_sharing_target_layer_name", None):
            continue

        spec = attn_module.get_kv_cache_spec(vllm_config)
        if spec is None:
            continue

        if isinstance(attn_module, MLAAttention):
            cache_sparse_sfa_c8 = False
            if getattr(attn_module.impl, "fa_quant_layer", False):
                head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                dtype, cache_dtype_str = attn_module.impl.dtype, None
            elif enable_sfa_quant(vllm_config):
                cache_sparse_sfa_c8 = bool(getattr(attn_module.impl, "enable_sparse_sfa_c8", False))
                if cache_sparse_sfa_c8:
                    head_size = get_sfa_qsfa_packed_head_dim(
                        vllm_config.model_config.hf_text_config.kv_lora_rank,
                        vllm_config.model_config.hf_text_config.qk_rope_head_dim,
                    )
                    dtype = c8_k_cache_dtype
                else:
                    head_size = (
                        vllm_config.model_config.hf_text_config.kv_lora_rank
                        + vllm_config.model_config.hf_text_config.qk_rope_head_dim
                    )
                    dtype = get_kv_cache_torch_dtype(
                        vllm_config.cache_config.cache_dtype,
                        vllm_config.model_config.dtype,
                    )
                cache_dtype_str = vllm_config.cache_config.cache_dtype
            else:
                head_size = spec.head_size
                dtype = spec.dtype
                cache_dtype_str = spec.cache_dtype_str
            spec = AscendMLAAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                cache_dtype_str=cache_dtype_str,
                cache_sparse_sfa_c8=cache_sparse_sfa_c8,
            )
        if isinstance(attn_module, DeepseekV32IndexerCache):
            cache_sparse_li_c8 = get_ascend_config().is_sparse_li_c8_layer(layer_name)
            kv_cache_spec[layer_name] = AscendSFAIndexerCacheSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=vllm_config.model_config.hf_text_config.index_head_dim,
                dtype=c8_k_cache_dtype
                if cache_sparse_li_c8
                else get_kv_cache_torch_dtype(
                    vllm_config.cache_config.cache_dtype,
                    vllm_config.model_config.dtype,
                ),
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                scale_dim=1 if cache_sparse_li_c8 else 0,
                scale_dtype=c8_k_scale_cache_dtype if cache_sparse_li_c8 else torch.int8,
                cache_sparse_li_c8=cache_sparse_li_c8,
            )
            continue

        kv_cache_spec[layer_name] = spec

    return kv_cache_spec


        if enable_sfa_quant(vllm_config) and bool(getattr(example_spec, "cache_sparse_sfa_c8", False)):
            k_size = kv_cache_tensor.size
            k_tensor = _allocate_int8_cache_tensor(k_size, alignment, device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = (k_tensor,)
        else:
            k_dim, v_dim = _get_attention_kv_cache_dims(example_layer_name, example_spec)
            if enable_fa_quant(vllm_config):
                k_factor, v_factor = vllm_config.quant_config.get_kv_quant_split_factor(
                    example_layer_name, [k_dim, v_dim]
                )
            else:
                k_factor, v_factor = calc_split_factor([k_dim, v_dim])
            k_size = int(kv_cache_tensor.size // k_factor)
            v_size = int(kv_cache_tensor.size // v_factor)
            k_tensor = _allocate_int8_cache_tensor(k_size, alignment, device)
            v_tensor = _allocate_int8_cache_tensor(v_size, alignment, device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)




            sparse_sfa_c8 = enable_sfa_quant(vllm_config) and bool(getattr(kv_cache_spec, "cache_sparse_sfa_c8", False))
            if sparse_sfa_c8:
                (raw_k_tensor,) = raw_cache
                raw_v_tensor = None
                assert raw_k_tensor is not None
                total_bytes = raw_k_tensor.numel()
            else:
                if not isinstance(raw_cache, tuple):
                    raise ValueError(f"KV cache for {layer_name} must contain K and V tensors.")
                raw_k_tensor, raw_v_tensor = raw_cache
                total_bytes = raw_k_tensor.numel() + raw_v_tensor.numel()
            if total_bytes % kv_cache_spec.page_size_bytes:
                raise ValueError(f"KV cache for {layer_name} is not a whole number of pages.")
            num_blocks = total_bytes // kv_cache_spec.page_size_bytes
            num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
            kernel_num_blocks = num_blocks * num_blocks_per_kv_block
            kv_cache_shape = group.backend.get_kv_cache_shape(
                kernel_num_blocks,
                kernel_block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype,
            )

            if isinstance(kv_cache_spec, (AscendMLAAttentionSpec, MLAAttentionSpec)):
                num_blocks_, block_size_, num_kv_heads, _ = kv_cache_shape
                k_dim, v_dim = _get_attention_kv_cache_dims(layer_name, kv_cache_spec)
                k_shape = (num_blocks_, block_size_, num_kv_heads, k_dim)
                if sparse_sfa_c8:
                    k_shape = (num_blocks_, block_size_, num_kv_heads, kv_cache_spec.head_size)
                    v_dim = 0
                v_shape = (num_blocks_, block_size_, num_kv_heads, v_dim)
            else:
                k_shape = kv_cache_shape[1:]
                v_shape = (
                    *kv_cache_shape[1:-1],
                    getattr(kv_cache_spec, "head_size_v", kv_cache_spec.head_size),
                )

            k_dtype = v_dtype = kv_cache_spec.dtype
            if enable_fa_quant(vllm_config):
                k_dtype, v_dtype = vllm_config.quant_config.get_kv_quant_dtype(
                    layer_name,
                    kv_cache_spec.dtype,
                    vllm_config.model_config,
                )
            if sparse_sfa_c8:
                k_dtype = torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else torch.int8
                k_cache = raw_k_tensor.view(k_dtype).view(k_shape)
                kv_caches[layer_name] = (k_cache,)
            else:
                assert raw_v_tensor is not None
                k_cache = raw_k_tensor.view(k_dtype).view(k_shape)
                v_cache = raw_v_tensor.view(v_dtype).view(v_shape)
                kv_caches[layer_name] = (k_cache, v_cache)

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]
    return kv_caches

