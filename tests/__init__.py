    enable_sparse_li_c4: bool = False


    _sparse_li_c4_layer_ids: set[int] = dataclasses.field(default_factory=set, init=False, repr=False)
    _sparse_li_c4_layer_names: set[str] = dataclasses.field(default_factory=set, init=False, repr=False)
    _sparse_li_layer_filter_enabled: bool = dataclasses.field(default=False, init=False, repr=False)



        self.enable_sparse_li_c4 = self.enable_sparse_li_c4 and use_sparse
        if self.enable_sparse_li_c8 and self.enable_sparse_li_c4:
            raise ValueError("enable_sparse_li_c8 and enable_sparse_li_c4 are mutually exclusive.")


        ) = self._parse_sparse_li_layers_from_quant_config(
            quant_config, ("INT8_DYNAMIC", "W8A8_MXFP8"))
        (
            self._sparse_li_c4_layer_ids,
            self._sparse_li_c4_layer_names,
        ) = self._parse_sparse_li_layers_from_quant_config(
            quant_config, ("W4A4_MXFP4",))
        self._sparse_li_layer_filter_enabled = self._has_sparse_li_layer_config(quant_config)


    def _has_sparse_li_layer_config(quant_config: Any) -> bool:


    def _parse_sparse_li_layers_from_quant_config(
        cls, quant_config: Any, valid_quant_types: tuple[str, ...]
    ) -> tuple[set[int], set[str]]:

        VALID_QUANT_TYPES = ("INT8_DYNAMIC", "W8A8_MXFP8")删掉
            if matched_suffix is None or value not in valid_quant_types:



    @staticmethod
    def _is_sparse_li_layer(
        layer_name: str | None,
        enable_flag: bool,
        filter_enabled: bool,
        layer_names: set[str],
        layer_ids: set[int],
    ) -> bool:
        if not enable_flag:
            return False
        if not filter_enabled:
            return True
        if layer_name is None:
            return False

        normalized_layer_name = layer_name.rstrip(".")
        if any(
            normalized_layer_name == candidate or normalized_layer_name.startswith(f"{candidate}.")
            for candidate in layer_names
        ):
            return True
        from vllm.model_executor.models.utils import extract_layer_index

        ids = {extract_layer_index(normalized_layer_name)}
        return any(layer_id in layer_ids for layer_id in ids)

    def is_sparse_li_c8_layer(self, layer_name: str | None) -> bool:
        return self._is_sparse_li_layer(
            layer_name,
            self.enable_sparse_li_c8,
            self._sparse_li_layer_filter_enabled,
            self._sparse_li_c8_layer_names,
            self._sparse_li_c8_layer_ids,
        )

    def is_sparse_li_c4_layer(self, layer_name: str | None) -> bool:
        return self._is_sparse_li_layer(
            layer_name,
            self.enable_sparse_li_c4,
            self._sparse_li_layer_filter_enabled,
            self._sparse_li_c4_layer_names,
            self._sparse_li_c4_layer_ids,
        )

        "_sparse_li_c4_layer_ids",
        "_sparse_li_c4_layer_names",
        "_sparse_li_layer_filter_enabled",

        self.enable_sparse_li_c4 = get_ascend_config().is_sparse_li_c4_layer(self.k_cache.prefix)
        if self.enable_sparse_li_c4:
            self.c4_k_cache_dtype, self.c4_k_scale_cache_dtype = torch.uint8, torch.float8_e8m0fnu
            self.c4_k_op_dtype = torch_npu.float4_e2m1fn_x2


    @property
    def enable_sparse_li_quant(self) -> bool:
        return self.enable_sparse_li_c8 or self.enable_sparse_li_c4


两个
        if self.enable_sparse_li_quant and AscendSFAIndexerBackend.q_hadamard is None:


    @property
    def num_cache_tensors(self) -> int:
        """Number of tensors this indexer's cache occupies in the composed
        ``kv_cache`` tuple (k cache only, or k cache plus scale cache)."""
        return 2 if self.enable_sparse_li_quant else 1

    def _quantize_li_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Hadamard transform and quantize for LI C8 or C4 path."""
        x = x @ AscendSFAIndexerBackend.q_hadamard
        shape_ori = x.shape
        x = x.view(-1, self.head_dim)
        if self.enable_sparse_li_c4:
            x, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=self.c4_k_op_dtype)
            scale = scale.view(*shape_ori[:-1], *scale.shape[-2:])
            return x, scale.view(self.c4_k_scale_cache_dtype)
        x, scale = torch_npu.npu_dynamic_quant(x, dst_type=self.c8_k_cache_dtype)
        return x, scale.to(self.c8_k_scale_cache_dtype)

    def write_cache(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        indexer_attn_metadata: Any | None = None,
    ) -> None:
        """Persist ``k_li`` (and ``k_li_scale`` when LI C8 is enabled) into
        this indexer's own cache tensors: slot 0 of ``self.k_cache.kv_cache``
        is the k cache, slot 1 (present only for LI C8) is the scale cache.

        ``forward`` calls this after ``_gather_cache_inputs`` has resolved
        the parallel layout of the tensors and the slot mapping; variants
        with a different cache layout should override it.
        ``indexer_attn_metadata`` is this indexer's own layer metadata; the
        LI C8 reshape-optim path reads its group fields.
        """
        indexer_k_cache = self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT]
        use_reshape_optim = self._use_c8_reshape_optim()

        def _store_kv(tensor: torch.Tensor, cache: torch.Tensor) -> None:
            if use_reshape_optim:
                assert indexer_attn_metadata is not None
                torch.ops._C_ascend.store_kv_block(
                    tensor,
                    cache,
                    indexer_attn_metadata.group_len,
                    indexer_attn_metadata.group_key_idx,
                    indexer_attn_metadata.group_key_cache_idx,
                    indexer_attn_metadata.block_size,
                )
            else:
                torch_npu.npu_scatter_nd_update_(
                    cache.view(-1, tensor.shape[-1]),
                    slot_mapping.view(-1, 1),
                    tensor.view(-1, tensor.shape[-1]),
                )

        _store_kv(k_li, indexer_k_cache)
        if self.enable_sparse_li_quant:
            assert k_li_scale is not None
            indexer_scale_cache = self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT]
            # C4 scale is stored as float8_e8m0fnu; npu_scatter_nd_update_
            # requires uint8 view for non-standard dtypes.
            if self.enable_sparse_li_c4:
                k_li_scale = k_li_scale.view(torch.uint8)
                indexer_scale_cache = indexer_scale_cache.view(torch.uint8)
            _store_kv(k_li_scale, indexer_scale_cache)


        if self.enable_sparse_li_quant:
            k_li, k_li_scale = self._quantize_li_tensor(k_li)



            if self.enable_sparse_li_quant:


         if self.enable_sparse_li_quant:
            q_li_shape_ori = q_li.shape
            q_li, q_li_scale = self._quantize_li_tensor(q_li)

        return DeviceOperator.indexer_select_post_process(
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            self.k_cache.kv_cache,
            INDEXER_K_CACHE_SLOT,
            INDEXER_SCALE_CACHE_SLOT,
            indexer_metadata,
            indexer_metadata.actual_seq_lengths_query,
            indexer_metadata.actual_seq_lengths_key,
            self.enable_sparse_li_c8,
            self.enable_sparse_li_c4,
            self.use_torch_npu_lightning_indexer,
        )

        self.enable_sparse_li_c4 = self.has_indexer and self.indexer.enable_sparse_li_c4


    li_quant_mode: str = ""  # "" / "cache_sparse_li_c8" / "cache_sparse_li_c4"

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSFAIndexerCacheSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        dtype_set = set(spec.dtype for spec in specs)
        scale_dim_set = set(spec.scale_dim for spec in specs)
        scale_dtype_set = set(spec.scale_dtype for spec in specs)
        cache_li_quant_mode_set = set(spec.li_quant_mode for spec in specs)
        sfa_dcp_replicated_indexer_size_set = set(spec.sfa_dcp_replicated_indexer_size for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(dtype_set) == 1
            and len(scale_dim_set) == 1
            and len(scale_dtype_set) == 1
            and len(cache_li_quant_mode_set) == 1
            and len(sfa_dcp_replicated_indexer_size_set) == 1
        ), (
            "All SFA indexer cache layers in the same KV cache group must use "
            "the same dtype, scale layout, quantization method, LI quant mode "
            "and DCP replication size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=dtype_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            scale_dim=scale_dim_set.pop(),
            scale_dtype=scale_dtype_set.pop(),
            li_quant_mode=cache_li_quant_mode_set.pop(),
            sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size_set.pop(),
        )


        enable_sparse_li_c4: bool,


        if enable_sparse_li_c4:
            raise RuntimeError("C4 lightning indexer is only supported on A5 devices.")
        elif enable_sparse_li_c8:


        enable_sparse_li_c4: bool,

        if enable_sparse_li_c4:
            assert len(kv_cache) == 2
            assert q_li_shape_ori is not None
            assert q_li_scale is not None

            key_dequant_scale = kv_cache[indexer_scale_cache_idx]
            weights_c4 = weights.to(torch.float32)
            cu_seqlens_q = torch.cat([
                torch.zeros(1, dtype=actual_seq_lengths_query.dtype,
                            device=actual_seq_lengths_query.device),
                actual_seq_lengths_query,
            ])
            seqused_k = actual_seq_lengths_key
            num_heads_q = q_li_shape_ori[1]
            head_dim = q_li_shape_ori[-1]
            batch_size = seqused_k.shape[0]

            import vllm_ascend.vllm_ascend_C  # noqa: F401, PLC0415

            metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                num_heads_q=num_heads_q, num_heads_k=1, head_dim=head_dim,
                topk=2048, quant_mode=5,
                cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
                batch_size=batch_size, max_seqlen_q=-1, max_seqlen_k=-1,
                layout_q="TND", layout_k="PA_BBND",
                mask_mode=3, cmp_ratio=1, device=str(q_li.device),
            )
            q_li_packed_shape = (*q_li_shape_ori[:-1], q_li_shape_ori[-1] // 2)
            sparse_indices, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
                q_li.view(q_li_packed_shape), kv_cache[indexer_cache_idx], weights_c4,
                q_li_scale, key_dequant_scale,
                topk=2048, quant_mode=5,
                cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
                block_table=attn_metadata.block_table, metadata=metadata,
                max_seqlen_q=-1, layout_q="TND", layout_k="PA_BBND",
                mask_mode=3, cmp_ratio=1, return_value=0,
            )
            topk_indices = sparse_indices
        elif enable_sparse_li_c8:
            # ``kv_cache`` is the indexer's own cache tuple (k + scale).
            assert len(kv_cache) == 2
            assert q_li_shape_ori is not None

            q_li_scale = q_li_scale.view(q_li_shape_ori[:-1])
            key_dequant_scale = kv_cache[indexer_scale_cache_idx].squeeze(2)

            topk_indices = torch_npu.npu_quant_lightning_indexer(
                query=q_li.view(q_li_shape_ori),
                key=kv_cache[indexer_cache_idx],
                weights=weights,
                query_dequant_scale=q_li_scale,
                key_dequant_scale=key_dequant_scale,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        else:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q_li.view(q_li_shape_ori) if q_li_shape_ori is not None else q_li,
                key=kv_cache[indexer_cache_idx],
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=attn_metadata.block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        return topk_indices


        self.enable_sparse_li_c4 = self.ascend_config.enable_sparse_li_c4

        if self.enable_sparse_li_c4:
            self.c4_k_cache_dtype = torch.uint8
            self.c4_k_scale_cache_dtype = torch.float8_e8m0fnu


                        if current_kv_cache_spec.li_quant_mode == "cache_sparse_li_c4":
                            indexer_scale_cache_shape = (*indexer_scale_cache_shape[:-1],
                                current_kv_cache_spec.head_size * 2 // 64, 2)



            elif isinstance(attn_module, DeepseekV32IndexerCache):
                # TODO: This mirrors upstream's separated KV/indexer specs for
                # SFA, but keeps Ascend-specific shape/block-size accounting.
                # Remove this special case once the generic vLLM spec/backend
                # path can describe the Ascend SFA indexer layout directly.
                cache_sparse_li_c8 = self.ascend_config.is_sparse_li_c8_layer(layer_name)
                cache_sparse_li_c4 = self.ascend_config.is_sparse_li_c4_layer(layer_name)
                head_dim = self.model_config.hf_text_config.index_head_dim
                kv_cache_spec[layer_name] = AscendSFAIndexerCacheSpec(
                    block_size=self.block_size,
                    num_kv_heads=1,
                    head_size=head_dim // 2 if cache_sparse_li_c4 else head_dim,
                    dtype=self.c4_k_cache_dtype if cache_sparse_li_c4
                    else self.c8_k_cache_dtype if cache_sparse_li_c8
                    else self.kv_cache_dtype,
                    cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
                    scale_dim=head_dim // 64 * 2 if cache_sparse_li_c4 else 1 if cache_sparse_li_c8 else 0,
                    scale_dtype=self.c4_k_scale_cache_dtype if cache_sparse_li_c4
                    else self.c8_k_scale_cache_dtype if cache_sparse_li_c8 else torch.int8,
                    li_quant_mode="cache_sparse_li_c4" if cache_sparse_li_c4
                    else "cache_sparse_li_c8" if cache_sparse_li_c8 else "",
                    sfa_dcp_replicated_indexer_size=self.sfa_dcp_replicated_indexer_size,
                )

    ascend_config = get_ascend_config()
    if ascend_config.enable_sparse_li_c4:
        c4_k_cache_dtype = torch.uint8
        c4_k_scale_cache_dtype = torch.float8_e8m0fnu



        if isinstance(attn_module, DeepseekV32IndexerCache):
            cache_sparse_li_c8 = ascend_config.is_sparse_li_c8_layer(layer_name)
            cache_sparse_li_c4 = ascend_config.is_sparse_li_c4_layer(layer_name)
            head_dim = vllm_config.model_config.hf_text_config.index_head_dim
            kv_cache_spec[layer_name] = AscendSFAIndexerCacheSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=head_dim // 2 if cache_sparse_li_c4 else head_dim,
                dtype=c4_k_cache_dtype if cache_sparse_li_c4
                else c8_k_cache_dtype if cache_sparse_li_c8
                else get_kv_cache_torch_dtype(
                    vllm_config.cache_config.cache_dtype,
                    vllm_config.model_config.dtype,
                ),
                cache_dtype_str=vllm_config.cache_config.cache_dtype,
                scale_dim=head_dim // 64 * 2 if cache_sparse_li_c4 else 1 if cache_sparse_li_c8 else 0,
                scale_dtype=c4_k_scale_cache_dtype if cache_sparse_li_c4
                else c8_k_scale_cache_dtype if cache_sparse_li_c8 else torch.int8,
                li_quant_mode="cache_sparse_li_c4" if cache_sparse_li_c4
                else "cache_sparse_li_c8" if cache_sparse_li_c8 else "",
                sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size,
            )
            continue


                    if group_spec.li_quant_mode == "cache_sparse_li_c4":
                        indexer_scale_cache_shape = (*indexer_scale_cache_shape[:-1],
                            group_spec.head_size * 2 // 64, 2)

