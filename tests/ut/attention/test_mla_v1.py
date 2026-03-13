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




