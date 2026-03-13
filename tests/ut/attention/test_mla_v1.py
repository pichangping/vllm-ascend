from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    ascend_chunked_prefill_workspace_size,
    enable_cp,
    enabling_fa_quant,
    enabling_mlapo,
    maybe_save_kv_layer_to_connector,
    split_decodes_and_prefills,
    trans_rope_weight,
    transdata,
    wait_for_kv_layer_from_connector,
)
Co-authored-by: kunpengW-code <1289706727@qq.com>
Co-authored-by: linsheng1 <1950916997@qq.com>

def enabling_fa_quant(vllm_config: VllmConfig, layer_name) -> bool:
    is_decode_instance = (
        vllm_config.kv_transfer_config is not None
        and vllm_config.kv_transfer_config.is_kv_consumer
        and not vllm_config.kv_transfer_config.is_kv_producer
    )
    quant_config = vllm_config.quant_config
    enable_fa_quant = quant_config.enable_fa_quant if quant_config is not None else False
    fa_quant_layer = False
    if is_decode_instance and enable_fa_quant:
        id = "".join(re.findall(r"\.(\d+)\.", layer_name))
        if int(id) in quant_config.kvcache_quant_layers:
            fa_quant_layer = True
    return fa_quant_layer

from vllm_ascend.quantization.methods.w8a8_static import AscendW8A8LinearMethod


    dequant_scale_q_nope: torch.Tensor | None = None


        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.layer_name = kwargs.get("layer_name")
        self.fa_quant_layer = enabling_fa_quant(self.vllm_config, self.layer_name)
        self.dtype = torch.int8 if self.fa_quant_layer else self.vllm_config.model_config.dtype

                    dequant_scale_q_nope,
                    fak_descale_float,

                if dequant_scale_q_nope is None:
                    torch_npu.npu_fused_infer_attention_score.out(
                        q_nope,
                        k_nope,
                        k_nope,
                        query_rope=q_pe,
                        key_rope=k_pe,
                        num_heads=num_heads,
                        num_key_value_heads=num_kv_heads,
                        input_layout=input_layout,
                        atten_mask=attn_mask,
                        sparse_mode=sparse_mode,
                        scale=scale,
                        antiquant_mode=0,
                        antiquant_scale=None,
                        block_table=block_table,
                        block_size=block_size,
                        actual_seq_lengths_kv=seq_lens_list,
                        actual_seq_lengths=actual_seq_lengths,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                else:
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        q_nope,
                        k_nope,
                        k_nope,
                        query_rope=q_pe,
                        key_rope=k_pe,
                        num_query_heads=num_heads,
                        num_key_value_heads=num_kv_heads,
                        input_layout=input_layout,
                        atten_mask=attn_mask,
                        sparse_mode=sparse_mode,
                        softmax_scale=scale,
                        query_quant_mode=3,
                        key_quant_mode=0,
                        value_quant_mode=0,
                        dequant_scale_query=dequant_scale_q_nope,
                        dequant_scale_key=fak_descale_float,
                        dequant_scale_value=fak_descale_float,
                        block_table=block_table,
                        block_size=block_size,
                        actual_seq_kvlen=seq_lens_list,
                        actual_seq_qlen=actual_seq_lengths,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )

        elif self.fa_quant_layer:
            self._process_weights_for_fused_fa_quant()

    def _process_weights_for_fused_fa_quant(self):
        self.gamma1 = self.q_a_layernorm.weight.data  # type: ignore[union-attr]
        self.gamma2 = self.kv_a_layernorm.weight.data  # type: ignore[union-attr]

        wu_q = self.q_proj.weight.data

        self.wu_q = wu_q

        q_a_proj_fa3 = self.fused_qkv_a_proj.weight.data[..., : self.q_lora_rank].contiguous()  # type: ignore[union-attr]

        self.wd_q = q_a_proj_fa3

        kv_a_proj_fa3 = self.fused_qkv_a_proj.weight.data[..., self.q_lora_rank :].contiguous()  # type: ignore[union-attr]

        self.wd_kv = kv_a_proj_fa3

        self.dequant_scale_w_uq_qr = self.q_proj.weight_scale.data.view(1, -1).to(torch.float)
        q_a_proj_deq_scl = self.fused_qkv_a_proj.weight_scale[: self.q_lora_rank].contiguous()  # type: ignore[union-attr]
        self.dequant_scale_w_dq = q_a_proj_deq_scl.view(1, -1).to(torch.float)
        kv_a_proj_deq_scl = self.fused_qkv_a_proj.weight_scale[self.q_lora_rank :].contiguous()  # type: ignore[union-attr]
        self.dequant_scale_w_dkv_kr = kv_a_proj_deq_scl.view(1, -1).to(torch.float)

        layer = self.vllm_config.compilation_config.static_forward_context[self.layer_name]
        self.quant_kscale = layer.quant_kscale
        self.fak_descale_float = layer.fak_descale_float

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
        dequant_scale_q_nope=None,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = q_nope.size(0)
        # shape of knope/k_pe for npu graph mode should be:
        # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
        actual_seq_lengths = None
        if self.fa_quant_layer:
            nz_fmt_last_dim = 16
            k_nope = k_nope.view(
                -1, self.num_kv_heads, self.kv_lora_rank // (nz_fmt_last_dim * 2), block_size, nz_fmt_last_dim * 2
            )
            k_pe = k_pe.view(
                -1, self.num_kv_heads, self.qk_rope_head_dim // nz_fmt_last_dim, block_size, nz_fmt_last_dim
            )
        elif self.enable_kv_nz:

        elif self.fa_quant_layer:
            attn_mask = None
            input_layout = "BSND_NBSD"
            q_nope = q_nope.view(num_tokens, 1, self.num_heads, -1).contiguous()
            q_pe = q_pe.view(num_tokens, 1, self.num_heads, -1).contiguous()
            dequant_scale_q_nope = dequant_scale_q_nope.view(num_tokens, 1, self.num_heads)
            sparse_mode = 0
            actual_seq_lengths = None
            common_kwargs_v2 = {
                "query_rope": q_pe,
                "key_rope": k_pe,
                "num_query_heads": self.num_heads,
                "num_key_value_heads": self.num_kv_heads,
                "input_layout": input_layout,
                "atten_mask": attn_mask,
                "sparse_mode": sparse_mode,
                "softmax_scale": self.scale,
                "query_quant_mode": 3,
                "key_quant_mode": 0,
                "value_quant_mode": 0,
                "dequant_scale_query": dequant_scale_q_nope,
                "dequant_scale_key": self.fak_descale_float,
                "dequant_scale_value": self.fak_descale_float,
                "block_table": decode_meta.block_table,
                "block_size": block_size,
                "actual_seq_qlen": actual_seq_lengths,
                "actual_seq_kvlen": decode_meta.seq_lens_list,
            }
            attn_output_shape = (self.num_heads, num_tokens, 1, self.kv_lora_rank)


            attn_output = torch.empty(attn_output_shape, dtype=q_pe.dtype, device=q_pe.device)
            softmax_lse = torch.empty(num_tokens, dtype=q_pe.dtype, device=q_pe.device)
            attn_params = (
                weak_ref_tensors(q_nope),
                weak_ref_tensors(k_nope),
                weak_ref_tensors(q_pe),
                weak_ref_tensors(k_pe),
                self.num_heads,
                self.num_kv_heads,
                input_layout,
                weak_ref_tensors(attn_mask) if attn_mask is not None else None,
                sparse_mode,
                self.scale,
                decode_meta.block_table,
                block_size,
                decode_meta.seq_lens_list,
                actual_seq_lengths,
                weak_ref_tensors(attn_output),
                weak_ref_tensors(softmax_lse),
            )
            if self.fa_quant_layer:
                get_max_workspace_func = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace
                fused_infer_attention_func = torch_npu.npu_fused_infer_attention_score_v2.out
                attn_params = attn_params + (dequant_scale_q_nope, self.fak_descale_float)
                common_kwargs = common_kwargs_v2
            else:
                get_max_workspace_func = torch_npu._npu_fused_infer_attention_score_get_max_workspace
                fused_infer_attention_func = torch_npu.npu_fused_infer_attention_score.out
                attn_params = attn_params + (None, None)

            if workspace is None:
                workspace = get_max_workspace_func(q_nope, k_nope, k_nope, **common_kwargs)
                if forward_context.is_draft_model:
                    update_draft_graph_params_workspaces(num_tokens, workspace)
                else:
                    update_graph_params_workspaces(num_tokens, workspace)

            graph_params.attn_params[num_tokens].append(attn_params)

            torch.npu.graph_task_group_begin(stream)
            fused_infer_attention_func(
                q_nope, k_nope, k_nope, **common_kwargs, workspace=workspace, out=[attn_output, softmax_lse]
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        elif self.fa_quant_layer:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(q_nope, k_nope, k_nope, **common_kwargs_v2)
        else:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(q_nope, k_nope, k_nope, **common_kwargs)


        dequant_scale_q_nope = None
        if self.fa_quant_layer:
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope = torch_npu.npu_mla_prolog_v2(
                quantized_x,
                self.wd_q,
                self.wu_q,
                self.W_UK_T,
                self.wd_kv,
                self.gamma1,
                self.gamma2,
                sin,
                cos,
                attn_metadata.slot_mapping[:bsz].to(torch.int64),
                decode_k_nope,
                decode_k_pe,
                dequant_scale_x=pertoken_scale.view(-1, 1),
                dequant_scale_w_dq=self.dequant_scale_w_dq,
                dequant_scale_w_uq_qr=self.dequant_scale_w_uq_qr,
                dequant_scale_w_dkv_kr=self.dequant_scale_w_dkv_kr,
                quant_scale_ckv=self.quant_kscale,
                cache_mode="PA_NZ",
            )
        else:
            decode_q_nope = torch.empty(
                (hidden_states.shape[0], self.W_UK_T.shape[0], decode_k_nope.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            decode_q_pe = torch.empty(
                (hidden_states.shape[0], self.W_UK_T.shape[0], decode_k_pe.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            torch.ops._C_ascend.mla_preprocess(
                hidden_states,
                self.wd_qkv,
                self.deq_scale_qkv,
                self.gamma1,
                self.beta1,
                self.wu_q,
                self.qb_deq_scl,
                self.gamma2,
                cos,
                sin,
                self.W_UK_T,
                decode_k_nope,
                decode_k_pe,
                attn_metadata.slot_mapping[:bsz],
                quant_scale0=self.quant_scale0,
                quant_offset0=self.quant_offset0,
                bias0=self.quant_bias_qkv,
                quant_scale1=self.quant_scale1,
                quant_offset1=self.quant_offset1,
                bias1=self.qb_qt_bias,
                ctkv_scale=self.ctkv_scale,
                q_nope_scale=self.q_nope_scale,
                cache_mode="nzcache" if self.enable_kv_nz else "krope_ctkv",
                quant_mode="per_tensor_quant_asymm",
                q_out0=decode_q_nope,
                kv_cache_out0=decode_k_nope,
                q_out1=decode_q_pe,
                kv_cache_out1=decode_k_pe,
                enable_inner_out=False,
                inner_out=torch.tensor([], device=hidden_states.device),
            )
            decode_q_nope = decode_q_nope.view(bsz, self.num_heads, self.kv_lora_rank)
            decode_q_pe = decode_q_pe.view(bsz, self.num_heads, -1)

        decode_q_nope, decode_q_pe = self.reorg_decode_q(decode_q_nope, decode_q_pe)

        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope=dequant_scale_q_nope
        )
      
        if self.fa_quant_layer or (self.enable_mlapo and attn_metadata.num_decode_tokens <= MLAPO_MAX_SUPPORTED_TOKENS):
          
                decode_preprocess_res.dequant_scale_q_nope,


mla.py
            layer_name=f"{prefix}.attn",

import vllm_ascend.patch.worker.patch_weight_utils  # noqa



import logging
import sys

from vllm.model_executor.model_loader.weight_utils import maybe_remap_kv_scale_name

logger = logging.getLogger(__name__)


class ImportPatchDecorator:
    """Import patch decorator"""

    _patches = {}

    @classmethod
    def register(cls, module_name):
        """Decorator for registering module patches"""

        def decorator(func):
            cls._patches[module_name] = func
            return func

        return decorator

    @classmethod
    def apply_patches(cls):
        """Apply all patches"""
        for module_name, patch_func in cls._patches.items():
            if module_name in sys.modules:
                module = sys.modules[module_name]
                try:
                    patch_func(module)
                except Exception as e:
                    logger.error(f"Patch application failed {module_name}: {e}")


# 使用装饰器注册补丁
@ImportPatchDecorator.register("vllm.model_executor.models.deepseek_v2")
def patch_deepseek(module):
    ori_maybe_remap_kv_scale_name = maybe_remap_kv_scale_name

    def new_remap(name: str, params_dict: dict):
        name = ori_maybe_remap_kv_scale_name(name, params_dict)

        replace_scale_names = ["fa_q.scale", "fa_k.scale", "fa_v.scale", "fa_q.offset", "fa_k.offset", "fa_v.offset"]

        for scale_name in replace_scale_names:
            if name.endswith(scale_name):
                remap_name = name.replace(scale_name, f"mla_attn.mla_attn.{scale_name}")
                if remap_name in params_dict:
                    return remap_name
                else:
                    return remap_name.replace(".mla_attn", "")

        return name

    if hasattr(module, "maybe_remap_kv_scale_name"):
        module._original_maybe_remap_kv_scale_name = module.maybe_remap_kv_scale_name
        module.maybe_remap_kv_scale_name = new_remap


@ImportPatchDecorator.register("vllm.model_executor.model_loader.weight_utils")
def patch_weight_utils(module):
    if "vllm.model_executor.models.deepseek_v2" in sys.modules:
        deepseek = sys.modules["vllm.model_executor.models.deepseek_v2"]
        if hasattr(deepseek, "maybe_remap_kv_scale_name"):
            module.maybe_remap_kv_scale_name = deepseek.maybe_remap_kv_scale_name


original_import = __builtins__["__import__"]


def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)

    if name in ImportPatchDecorator._patches:
        try:
            ImportPatchDecorator._patches[name](module)
        except Exception as e:
            logger.error(f"Patch application failed during import {name}: {e}")

    return module


__builtins__["__import__"] = patched_import

ImportPatchDecorator.apply_patches()



import glob
import json
import os
import re

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase



MODELSLIM_CONFIG_FILENAME = "quant_model_description.json"

    def __init__(self, quant_config: dict[str, Any] | None = None):
        super().__init__()
        self.quant_description = quant_config if quant_config is not None else {}
        # TODO(whx): remove this adaptation after adding "shared_head"
        # to prefix of DeepSeekShareHead in vLLM.
        extra_quant_dict = {}
        for k in self.quant_description:
            if "shared_head" in k:
                new_k = k.replace(".shared_head.", ".")
                extra_quant_dict[new_k] = self.quant_description[k]
            if "weight_packed" in k:
                new_k = k.replace("weight_packed", "weight")
                extra_quant_dict[new_k] = self.quant_description[k]
        self.quant_description.update(extra_quant_dict)
        # Initialize attributes for type checking
        self.model_type: str | None = None
        self.hf_to_vllm_mapper: WeightsMapper | None = None
        self.vllm_to_hf_mapper: WeightsMapper | None = None
        self._apply_extra_quant_adaptations()

        elif isinstance(layer, AttentionLayerBase) and self.is_fa_quant_layer(prefix):


    def is_fa_quant_layer(self, prefix):
        if self.enable_fa_quant:
            _id = int("".join(re.findall(r"\.(\d+)\.", prefix)))
            if _id in self.kvcache_quant_layers:
                return True
        return False

    def maybe_update_config(self, model_name: str) -> None:
        """Load the ModelSlim quantization config from model directory.

        This method is called by vllm after get_quant_config() returns
        successfully. Since we return an empty list from get_config_filenames()
        to bypass vllm's built-in file lookup, we do the actual config loading
        here and provide user-friendly error messages when the config is missing.

        Args:
            model_name: Path to the model directory or model name.
        """
        # If quant_description is already populated (e.g. from from_config()),
        # there is nothing to do.
        if self.quant_description:
            return

        # Try to find and load the ModelSlim config file
        if os.path.isdir(model_name):
            config_path = os.path.join(model_name, MODELSLIM_CONFIG_FILENAME)
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    self.quant_description = json.load(f)
                self._apply_extra_quant_adaptations()
                self._add_kvcache_quant_metadata()
                return

            # Check if there are any json files at all to help diagnose
            json_files = glob.glob(os.path.join(model_name, "*.json"))
            json_names = [os.path.basename(f) for f in json_files]
        else:
            json_names = []

        # Config file not found - raise a friendly error message
        raise ValueError(
            "\n"
            + "=" * 80
            + "\n"
            + "ERROR: ModelSlim Quantization Config Not Found\n"
            + "=" * 80
            + "\n"
            + "\n"
            + f"You have enabled '--quantization {ASCEND_QUANTIZATION_METHOD}' "
            + "(ModelSlim quantization),\n"
            + f"but the model at '{model_name}' does not contain the required\n"
            + f"quantization config file ('{MODELSLIM_CONFIG_FILENAME}').\n"
            + "\n"
            + "This usually means the model weights are NOT quantized by "
            + "ModelSlim.\n"
            + "\n"
            + "Please choose one of the following solutions:\n"
            + "\n"
            + "  Solution 1: Remove the quantization option "
            + "(for float/unquantized models)\n"
            + "  "
            + "-" * 58
            + "\n"
            + f"    Remove '--quantization {ASCEND_QUANTIZATION_METHOD}' from "
            + "your command if you want to\n"
            + "    run the model with the original (float) weights.\n"
            + "\n"
            + "    Example:\n"
            + f"      vllm serve {model_name}\n"
            + "\n"
            + "  Solution 2: Quantize your model weights with ModelSlim first\n"
            + "  "
            + "-" * 58
            + "\n"
            + "    Use the ModelSlim tool to quantize your model weights "
            + "before deployment.\n"
            + "    After quantization, the model directory should contain "
            + f"'{MODELSLIM_CONFIG_FILENAME}'.\n"
            + "    For more information, please refer to:\n"
            + "    https://gitee.com/ascend/msit/tree/master/msmodelslim\n"
            + "\n"
            + (f"  (Found JSON files in model directory: {json_names})\n" if json_names else "")
            + "=" * 80
        )

    def _apply_extra_quant_adaptations(self) -> None:
        """Apply extra adaptations to the quant_description dict.

        This handles known key transformations such as shared_head and
        weight_packed mappings.
        """
        extra_quant_dict = {}
        for k in self.quant_description:
            if "shared_head" in k:
                new_k = k.replace(".shared_head.", ".")
                extra_quant_dict[new_k] = self.quant_description[k]
            if "weight_packed" in k:
                new_k = k.replace("weight_packed", "weight")
                extra_quant_dict[new_k] = self.quant_description[k]
        self.quant_description.update(extra_quant_dict)

    def get_scaled_act_names(self) -> list[str]:
        return []

    def _add_kvcache_quant_metadata(self):
        fa_quant_type = self.quant_description.get("fa_quant_type", "")
        self.enable_fa_quant = fa_quant_type != ""
        self.kvcache_quant_layers = []
        if self.enable_fa_quant:
            for key in self.quant_description:
                if "fa_k.scale" in key:
                    _id = "".join(re.findall(r"\.(\d+)\.", key))
                    self.kvcache_quant_layers.append(int(_id))

from .kv_c8 import AscendFAQuantAttentionMethod


        self.kvbytes = {}

                        if layer_name in self.kvbytes:
                            head_size = (
                                self.model_config.hf_text_config.qk_rope_head_dim * self.kvbytes[layer_name][1]
                                + self.model_config.hf_text_config.kv_lora_rank * self.kvbytes[layer_name][0]
                            )
                        else:
                            head_size = (
                                self.model_config.hf_text_config.qk_rope_head_dim
                                + self.model_config.hf_text_config.kv_lora_rank
                            )
                            
                    elif layer_name in self.kvbytes:
                        k_tensor_split_factor = head_size / (
                            self.model_config.hf_text_config.kv_lora_rank * self.kvbytes[layer_name][0]
                        )
                        v_tensor_split_factor = head_size / (
                            self.model_config.hf_text_config.qk_rope_head_dim * self.kvbytes[layer_name][1]
                        )

                    if layer_name in self.kvbytes:
                        k_cache = raw_k_tensor.view(dtype).view(k_shape)
                        v_cache = raw_v_tensor.view(self.vllm_config.model_config.dtype).view(v_shape)
                    else:
                        k_cache = raw_k_tensor.view(dtype).view(k_shape)
                        v_cache = raw_v_tensor.view(dtype).view(v_shape)


        def dtype_to_bytes(dtype: torch.dtype) -> int:
            """将 torch.dtype 转换为字节数"""
            return torch.tensor([], dtype=dtype).element_size()

                elif getattr(attn_module.impl, "fa_quant_layer", False):
                    block_size = self.vllm_config.cache_config.block_size
                    head_size = attn_module.head_size + attn_module.qk_rope_head_dim
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=head_size,
                        dtype=attn_module.impl.dtype,
                        cache_dtype_str=None,
                    )
                    if layer_name not in self.kvbytes:
                        self.kvbytes[layer_name] = [
                            dtype_to_bytes(attn_module.impl.dtype),
                            dtype_to_bytes(self.vllm_config.model_config.dtype),
                        ]
