import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "./LLM-Viewer"))
from utils import str_number, str_number_time
from model_analyzer import ModelAnalyzer
import math

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]

import pprint

def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key:30}: {value}")


class AdaptiveAnalyzer(ModelAnalyzer):
    def __init__(self, model_id, hardware, config_file=None, source="huggingface"):
        super().__init__(model_id, hardware, config_file, source)

    def analyze_one_layer(
        self,
        prompt_len,
        num_heads=None,
        batchsize=1,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1
    ):
        """
        prompt_len: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        assert prompt_len > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.prompt_len = prompt_len
        self.tp_size = tp_size

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        if num_heads is None:
            num_heads = num_attention_heads
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)

        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_q_out_proj = name in ["q_proj", "out_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                "decode",
                name,
                OPs=ic * oc * batchsize * 2 * num_heads // num_attention_heads,
                load_weight=ic * oc * w_byte * num_heads // num_attention_heads,
                load_act=ic * batchsize * a_byte * num_heads // num_attention_heads,
                store_act=0 if is_kv_proj else (oc * batchsize * a_byte * num_heads // num_attention_heads),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ic * oc * batchsize * prompt_len * 2 * num_heads // num_attention_heads,
                load_weight=ic * oc * w_byte * num_heads // num_attention_heads,
                load_act=ic * batchsize * prompt_len * a_byte * num_heads // num_attention_heads,
                store_act=0 if is_kv_proj else (oc * batchsize * prompt_len * a_byte * num_heads // num_attention_heads),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * prompt_len * kv_byte),
            )

        # for attention
        head_size = hidden_size // num_attention_heads
        # for decode
        qk_matmul_OPs = prompt_len * head_size * num_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * prompt_len * num_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * num_heads * prompt_len * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * num_heads * a_byte
            o_numel = 1 * prompt_len * batchsize * num_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (prompt_len) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * num_heads * a_byte,
                store_act=1 * prompt_len * batchsize * num_heads * a_byte,
                load_kv_cache=(prompt_len) * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * prompt_len * batchsize * num_heads) * a_byte,
                store_act=1 * head_size * batchsize * num_heads * a_byte,
                load_kv_cache=(prompt_len * head_size * batchsize * num_key_value_heads) * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_heads * prompt_len * 1 * a_byte,
                store_act=batchsize * num_heads * prompt_len * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in config.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 2 * num_heads // num_attention_heads,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte * 2 * num_heads // num_attention_heads,
                store_act=batchsize * hidden_size * 1 * a_byte * num_heads // num_attention_heads,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = prompt_len * prompt_len * head_size * num_heads * batchsize * 2
        sv_matmul_OPs = prompt_len * head_size * prompt_len * num_heads * batchsize * 2
        softmax_OPs = batchsize * num_heads * prompt_len * prompt_len * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(prompt_len / block_size_r)
            q_numel = prompt_len * head_size * batchsize * num_heads * a_byte
            o_numel = prompt_len * prompt_len * batchsize * num_heads * a_byte
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (prompt_len) * head_size * batchsize * num_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=prompt_len * head_size * batchsize * num_heads * a_byte,
                store_act=prompt_len * prompt_len * batchsize * num_heads * a_byte,
                load_kv_cache=prompt_len * head_size * batchsize * num_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=prompt_len * prompt_len * batchsize * num_heads * a_byte,
                store_act=prompt_len * head_size * batchsize * num_heads * a_byte,
                load_kv_cache=prompt_len * head_size * batchsize * num_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_heads * prompt_len * prompt_len * a_byte,
                store_act=batchsize * num_heads * prompt_len * prompt_len * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in config.get_norm_layers(model_params):
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * prompt_len * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * prompt_len * a_byte,
                store_act=batchsize * hidden_size * prompt_len * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * prompt_len * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * prompt_len * a_byte,
                store_act=batchsize * hidden_size * prompt_len * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * prompt_len * 1 * 2 * num_heads // num_attention_heads,
                load_weight=0,
                load_act=batchsize * hidden_size * prompt_len * a_byte * 2 * num_heads // num_attention_heads,
                store_act=batchsize * hidden_size * prompt_len * a_byte * num_heads // num_attention_heads,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        return self.results
    
    def analyze_all_layers(
        self,
        prompt_len,
        num_heads=[8, 12, 24],
        batchsize=1,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1
    ):
        results = []
        for curr_num_heads in num_heads:
            if curr_num_heads >=0:
                results.append(self.analyze_one_layer(prompt_len, 
                                                    curr_num_heads, 
                                                    batchsize,
                                                    w_bit,
                                                    a_bit,
                                                    kv_bit,
                                                    use_flashattention,
                                                    kv_token_ratio,
                                                    tp_size))
            
        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_results in results:
                for layer_name, result in layer_results[stage].items():
                    for data_name in ALL_DATA_NAMES:
                        total_results[stage][data_name] += result[data_name]

        # memory footprint
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = 0
        for layer_name, result in results[-1]["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = 0
        for layer_name, result in results[-1]["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # lm_head
        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        name = "lm_head"
        args = {"batchsize": batchsize, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]

        return total_results
    
    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        num_heads=[8, 12, 24],
        batchsize=1,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1
    ):
        # FLOPs are computed from prompt_len (LLM input sequence length). No fixed N is used.
        # For Exp1 (vision token prune), the wrapper passes the shortened prompt_len = text + K vision tokens.
        prefill_result = self.analyze_all_layers(
            prompt_len,
            num_heads,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size
        )
        prefill_time = inference_time = prefill_result["prefill"]["inference_time"]
        prefill_flops = flops = prefill_result["prefill"]["OPs"]
        prefill_memory_consumption = memory_consumption = prefill_result["prefill"]["memory_consumption"]

        for i in range(prompt_len, prompt_len + max(0, gen_len) - 1):
            result = self.analyze_all_layers(i, num_heads, batchsize, w_bit, a_bit, kv_bit, use_flashattention=use_flashattention, tp_size=tp_size)
            inference_time += result["decode"]["inference_time"]
            flops += result["decode"]["OPs"]
            memory_consumption += result["decode"]["memory_consumption"]
        # Avoid ZeroDivisionError when gen_len <= 0 (e.g. outputs.sequences only had prompt/generated part)
        avg_flops = (flops / gen_len) if gen_len > 0 else float(prefill_flops)

        return {"flops": flops,
                "avg_flops": avg_flops,
                "prefill_flops": prefill_flops,
                "prefill_time": prefill_time,
                "memory_consumption": memory_consumption,
                "prefill_memory_consumption": prefill_memory_consumption}