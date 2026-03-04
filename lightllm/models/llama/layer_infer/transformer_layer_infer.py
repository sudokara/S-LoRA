import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Dict, Any, List
import triton

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.models.lora_cpu import Lora_CPU
from lightllm.models.lora_gpu import Lora_GPU

import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)


class LlamaTransformerLayerInfer(TransformerLayerInferTpl):
    """
    """
    def __init__(self, layer_num: int, tp_rank: int, world_size: int,
                 network_config: Dict[str, Any], mode: List[str] = [],
                 gpu_lora: Lora_GPU = None, cpu_lora: Lora_CPU = None):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self.cpu_lora = cpu_lora
        self.gpu_lora = gpu_lora
        self.count_use_cpu_lora = 0
        self.count_use_gpu_lora = 0
        self._bind_func()
        return
    
    def _bind_func(self):
        if "ppl_int8kv" in self.mode:
            self._token_attention_kernel = self._token_decode_attention_ppl_int8kv
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_ppl_int8kv
        elif "triton_int8kv" in self.mode:
            self._token_attention_kernel = self._token_decode_attention_int8kv
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_int8kv
        elif "triton_flashdecoding" in self.mode:
            self._token_attention_kernel = self._token_decode_attention_flashdecoding
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_normal   
        else:
            self._token_attention_kernel = self._token_decode_attention_normal
            self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_normal   
        return
    
    def _att_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_)
    
    def _ffn_norm(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_)

    def _get_qkv(self, input, cache_k, cache_v, infer_state:LlamaInferStateInfo, 
                    layer_weight:LlamaTransformerLayerWeight, time_breakdown)->torch.Tensor:
        # layer_weight.q_weight_.shape: (self.embed_dim_, self.embed_dim_//world_size)
        # input shape: (batch_size*token_length, self.embed_dim_)
        # torch.cuda.synchronize()
        if "gpu_lora" in self.mode:
            time_breakdown["gpu_lora_layers"] = time_breakdown.get("gpu_lora_layers", 0) + 1
            gpu_invoke_start = time.time()
            q_lora = self.gpu_lora.invoke_one_layer_q( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_)
            q_lora = q_lora.view( -1, self.embed_dim_//self.world_size_ )
            k_lora = self.gpu_lora.invoke_one_layer_k( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_)
            k_lora = k_lora.view( -1, self.embed_dim_//self.world_size_ )
            v_lora = self.gpu_lora.invoke_one_layer_v( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_)
            v_lora = v_lora.view( -1, self.embed_dim_//self.world_size_ )
            gpu_invoke_end = time.time()
            time_breakdown["gpu_invoke"] = time_breakdown.get("gpu_invoke", 0) + (gpu_invoke_end - gpu_invoke_start)*1000
        elif "cpu_lora" in self.mode:
            time_breakdown["cpu_lora_layers"] = time_breakdown.get("cpu_lora_layers", 0) + 1
            cpu_invoke_start = time.time()
            self.cpu_lora.invoke_lora_async(input, input.shape[0]//infer_state.batch_size, self.layer_num_, None, is_o=False)
            cpu_invoke_end = time.time()
            time_breakdown["cpu_invoke_qkv"] = time_breakdown.get("cpu_invoke_qkv", 0) + (cpu_invoke_end - cpu_invoke_start)*1000
        elif "aaas" in self.mode or "aaas-gpu" in self.mode or "aaas-cached" in self.mode:
            if "gpu_lora" in infer_state.aaas_mode:
                gpu_lora_prefill = False
                if "prefill" in infer_state.aaas_mode:
                    gpu_lora_prefill = True
                time_breakdown["gpu_lora_layers"] = time_breakdown.get("gpu_lora_layers", 0) + 1
                gpu_invoke_start = time.time()
                if gpu_lora_prefill:
                    q_lora = self.gpu_lora.invoke_one_layer_q( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_, gpu_lora_prefill)
                    q_lora = q_lora.view( -1, self.embed_dim_//self.world_size_ )
                    k_lora = self.gpu_lora.invoke_one_layer_k( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_, gpu_lora_prefill)
                    k_lora = k_lora.view( -1, self.embed_dim_//self.world_size_ )
                    v_lora = self.gpu_lora.invoke_one_layer_v( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_, gpu_lora_prefill)
                    v_lora = v_lora.view( -1, self.embed_dim_//self.world_size_ )
                else:
                    q_lora = self.gpu_lora.invoke_one_layer_q( input, self.layer_num_, gpu_lora_prefill)
                    k_lora = self.gpu_lora.invoke_one_layer_k( input, self.layer_num_, gpu_lora_prefill)
                    v_lora = self.gpu_lora.invoke_one_layer_v( input, self.layer_num_, gpu_lora_prefill)
                gpu_invoke_end = time.time()
                time_breakdown["gpu_invoke"] = time_breakdown.get("gpu_invoke", 0) + (gpu_invoke_end - gpu_invoke_start)*1000
            elif infer_state.aaas_mode == "cpu_lora" or infer_state.aaas_mode == "cpu_lora_decode":
                # aaas_use_gpu_lora = (self.gpu_lora.use_gpu() == 0)
                aaas_use_gpu_lora = self.gpu_lora.use_gpu_pipeline( self.layer_num_ ) 
                if (self.count_use_gpu_lora + self.count_use_cpu_lora) % 50 == 0: # print counters for logging
                    logging.critical("Attention Layer {} use GPU, CPU: {},{}".format(self.layer_num_, self.count_use_gpu_lora, self.count_use_cpu_lora))
                if aaas_use_gpu_lora and infer_state.aaas_mode != "cpu_lora_decode":
                    self.count_use_gpu_lora += 1
                    time_breakdown["gpu_lora_layers"] = time_breakdown.get("gpu_lora_layers", 0) + 1
                    gpu_invoke_start = time.time()
                    q_lora = self.gpu_lora.invoke_one_layer_q( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_, True)
                    q_lora = q_lora.view( -1, self.embed_dim_//self.world_size_ )
                    k_lora = self.gpu_lora.invoke_one_layer_k( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_, True)
                    k_lora = k_lora.view( -1, self.embed_dim_//self.world_size_ )
                    v_lora = self.gpu_lora.invoke_one_layer_v( input.view(infer_state.batch_size, -1, self.embed_dim_), self.layer_num_, True)
                    v_lora = v_lora.view( -1, self.embed_dim_//self.world_size_ )
                    gpu_invoke_end = time.time()
                    time_breakdown["gpu_invoke"] = time_breakdown.get("gpu_invoke", 0) + (gpu_invoke_end - gpu_invoke_start)*1000
                else:
                    self.count_use_cpu_lora += 1
                    time_breakdown["cpu_lora_layers"] = time_breakdown.get("cpu_lora_layers", 0) + 1
                    cpu_invoke_start = time.time()
                    self.cpu_lora.invoke_lora_async(input, input.shape[0]//infer_state.batch_size, self.layer_num_, None, is_o=False)
                    cpu_invoke_end = time.time()
                    time_breakdown["cpu_invoke_qkv"] = time_breakdown.get("cpu_invoke_qkv", 0) + (cpu_invoke_end - cpu_invoke_start)*1000

                    # gpu2cpu_start = time.time()
                    # lora_input = input.view(infer_state.batch_size, -1, self.embed_dim_).to(device="cpu")
                    # gpu2cpu_end = time.time()
                    # time_breakdown["gpu2cpu_time"] = time_breakdown.get("gpu2cpu_time", 0) + 1000*(gpu2cpu_end - gpu2cpu_start)
                    # cpu_invoke_start = time.time()
                    # self.cpu_lora.invoke_lora_torch(lora_input, input.shape[0]//infer_state.batch_size, self.layer_num_, None)
                    # cpu_invoke_end = time.time()
                    # time_breakdown["cpu_invoke"] = time_breakdown.get("cpu_invoke", 0) + (cpu_invoke_end - cpu_invoke_start)*1000
        base_mm_start = time.time()
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_) # shape ( batch_size*token_length, self.embed_dim_//world_size)
        k = torch.mm(input.view(-1, self.embed_dim_), layer_weight.k_weight_)
        v = torch.mm(input.view(-1, self.embed_dim_), layer_weight.v_weight_)
        torch.cuda.synchronize()
        time_breakdown["base_qkv_matmul"] = time_breakdown.get("base_qkv_matmul", 0) + (time.time() - base_mm_start)*1000

        if "cpu_lora" in self.mode or (infer_state.aaas_mode == "cpu_lora" and not aaas_use_gpu_lora) or (infer_state.aaas_mode == "cpu_lora_decode"):
            collect_start = time.time()
            q_lora_res, k_lora_res, v_lora_res = self.cpu_lora.collect_qkv(input.shape[0]//infer_state.batch_size, infer_state.batch_size, None)
            collect_end = time.time()
            time_breakdown["collect"] = time_breakdown.get("collect", 0) + (collect_end - collect_start)*1000

            
            q_lora = q_lora_res.view(-1, q.shape[1])
            k_lora = k_lora_res.view(-1, k.shape[1])
            v_lora = v_lora_res.view(-1, v.shape[1])

        if "base" not in self.mode:   
            q += q_lora
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)

        if "base" not in self.mode:
            k += k_lora
        cache_k[:] = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        if "base" not in self.mode:
            v += v_lora
        cache_v[:] = v.view(-1, self.tp_v_head_num_, self.head_dim_)

        return q
    
    def _context_attention_kernel(self, q, k, v, infer_state:LlamaInferStateInfo, layer_weight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor

    def _get_o(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor

    def _ffn(self, input, infer_state:LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.up_proj)
        input = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None
        return ffn2_out
    
    def _copy_kv_to_mem_cache_normal(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_kv(key_buffer, mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(value_buffer, mem_index, mem_manager.value_buffer[self.layer_num_])
        return
    
    def _copy_kv_to_mem_cache_int8kv(self, key_buffer, value_buffer, mem_index, mem_manager):
        destindex_copy_quantize_kv(key_buffer,
                                    mem_index,
                                    mem_manager.key_buffer[self.layer_num_],
                                    mem_manager.key_scale_buffer[self.layer_num_])
        destindex_copy_quantize_kv(value_buffer,
                                    mem_index,
                                    mem_manager.value_buffer[self.layer_num_],
                                    mem_manager.value_scale_buffer[self.layer_num_])
        return
    
    def _copy_kv_to_mem_cache_ppl_int8kv(self, key_buffer, value_buffer, mem_index, mem_manager):
        from lightllm.models.llama.triton_kernel.ppl_quant_copy_kv import destindex_copy_quantize_kv
        destindex_copy_quantize_kv(key_buffer,
                                    mem_index,
                                    mem_manager.key_buffer[self.layer_num_],
                                    mem_manager.key_scale_buffer[self.layer_num_])
        destindex_copy_quantize_kv(value_buffer,
                                    mem_index,
                                    mem_manager.value_buffer[self.layer_num_],
                                    mem_manager.value_scale_buffer[self.layer_num_])
        return
    
    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.req_manager.req_to_token_indexs,
                      infer_state.b_req_idx,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)
        
        if triton.__version__ == "2.0.0":
            prob = torch.empty_like(att_m_tensor)
            token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
            att_m_tensor = None

            o_tensor = torch.empty_like(q)

            token_att_fwd2(prob,
                        infer_state.mem_manager.value_buffer[self.layer_num_],
                        o_tensor.view(calcu_shape1),
                        infer_state.req_manager.req_to_token_indexs,
                        infer_state.b_req_idx,
                        infer_state.b_start_loc,
                        infer_state.b_seq_len)
            prob = None
            return o_tensor
        elif triton.__version__ >= "2.1.0":
            o_tensor = torch.empty_like(q)
            from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
            token_softmax_reducev_fwd(att_m_tensor, 
                                      infer_state.mem_manager.value_buffer[self.layer_num_],
                                      o_tensor.view(calcu_shape1),
                                      infer_state.req_manager.req_to_token_indexs,
                                      infer_state.b_req_idx,
                                      infer_state.b_start_loc,
                                      infer_state.b_seq_len,
                                      infer_state.other_kv_index)
            return o_tensor
        else:
            raise Exception("not support triton version")

    def _token_decode_attention_int8kv(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")
        token_att_fwd_int8k(q.view(calcu_shape1),
                            infer_state.mem_manager.key_buffer[self.layer_num_],
                            infer_state.mem_manager.key_scale_buffer[self.layer_num_],
                            att_m_tensor,
                            infer_state.req_manager.req_to_token_indexs,
                            infer_state.b_req_idx,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)

        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)
        token_att_fwd2_int8v(prob,
                                infer_state.mem_manager.value_buffer[self.layer_num_],
                                infer_state.mem_manager.value_scale_buffer[self.layer_num_],
                                o_tensor.view(calcu_shape1),
                                infer_state.req_manager.req_to_token_indexs,
                                infer_state.b_req_idx,
                                infer_state.b_start_loc,
                                infer_state.b_seq_len,
                                infer_state.max_len_in_batch)
        prob = None
        return o_tensor
    
    def _token_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        from lightllm.models.llama.triton_kernel.flash_decoding import token_decode_attention_flash_decoding
        cache_k = infer_state.mem_manager.key_buffer[self.layer_num_]
        cache_v = infer_state.mem_manager.value_buffer[self.layer_num_]
        return token_decode_attention_flash_decoding(q, infer_state, self.tp_q_head_num_, self.head_dim_, cache_k, cache_v)
    
    def _token_decode_attention_ppl_int8kv(self, q, infer_state: LlamaInferStateInfo, layer_weight):
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        o_tensor = torch.empty_like(q)

        from lightllm_ppl_kernel import group8_int8kv_decode_attention
        # group_int8kv_decode_attention(at::Tensor o, at::Tensor q, at::Tensor k, at::Tensor k_s,  at::Tensor v,  at::Tensor v_s, at::Tensor b_loc, at::Tensor b_seq_len, int max_len_in_batch)
        group8_int8kv_decode_attention(o_tensor.view(calcu_shape1),
                                                          q.view(calcu_shape1),
                                                          infer_state.mem_manager.key_buffer[self.layer_num_],
                                                          infer_state.mem_manager.key_scale_buffer[self.layer_num_],
                                                          infer_state.mem_manager.value_buffer[self.layer_num_],
                                                          infer_state.mem_manager.value_scale_buffer[self.layer_num_],
                                                          infer_state.req_manager.req_to_token_indexs,
                                                          infer_state.b_req_idx,
                                                          infer_state.b_seq_len,
                                                          infer_state.max_len_in_batch)
           
        return o_tensor
