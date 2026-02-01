import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import logging
import os
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

# Per-sample messages (Exp1 / FLOPs) use logger.debug(). Enable via:
#   logging.getLogger("adallava.eval.adallava_wrapper").setLevel(logging.DEBUG)
#   or from scripts: DEBUG=1 bash scripts/eval/...
logger = logging.getLogger(__name__)
if os.environ.get("DEBUG", "").strip() in ("1", "true", "yes"):
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setLevel(logging.DEBUG)
        logger.addHandler(_h)

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models.llava import Llava
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from ..model.builder import load_pretrained_model
    from .ada_analyzer import AdaptiveAnalyzer
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("adallava")
class AdaLlava(Llava):
    """
    AdaLlava Model
    """

    def __init__(
        self,
        pretrained: str = "zhuoyanxu/ada-llava-v1.5-7b",
        hardware: str = "nvidia_V100",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        latency=1.0,
        token_budget=1.0,
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json
        **kwargs,
    ) -> None:
        super(Llava, self).__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        # Single-GPU: use "auto" so builder can set device_map to cuda:0 (same as demo), avoiding
        # tied-weights bug and vision tower on another device (cuda:1).
        _device_map = "auto" if (hasattr(accelerator, "num_processes") and accelerator.num_processes == 1) else self.device_map
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, device_map=_device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, device_map=_device_map, **llava_model_args)
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.latency = latency
        self.token_budget = token_budget
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self.analyzer = AdaptiveAnalyzer(pretrained, hardware, 'configs/Llama.py')

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            # prompts_input = contexts[0]

            question_input = []

            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # For adaptive vision token scheduler (Exp1), set token budget before encode/generate
            if getattr(self.model, "vision_token_controller", None) is not None:
                self.model.current_token_budget = self.token_budget
            try:
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    latency=self.latency,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    return_dict_in_generate=True,
                )
                cont = outputs.sequences
                prompt_len = getattr(outputs, "prompt_len", None)
                prompt_len_tokens = getattr(outputs, "prompt_len_tokens", None)
                gen_len = getattr(outputs, "gen_len", None)
                num_heads = getattr(outputs, "execution_plan", None)
                # Exp1: actual vision token count K after prune (for logging and FLOPs sanity-check).
                vision_tokens_K = getattr(self.model, "_last_vision_token_count", None)
                if vision_tokens_K is not None:
                    try:
                        vision_tokens_K = int(vision_tokens_K)
                    except Exception:
                        vision_tokens_K = None
                # Decode only the generated part; sanitize token IDs to avoid SentencePiece "piece id is out of range"
                vocab_size = len(self.tokenizer)
                eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
                pad_id = getattr(self.tokenizer, "pad_token_id", None)

                def _safe_id(i: int) -> int:
                    if not (0 <= i < vocab_size):
                        return eos_id
                    if i == 0 and pad_id is None:
                        return eos_id
                    return i

                batch_size = cont.shape[0]
                seq_len = cont.shape[1]
                # Slice by token-space prompt length (outputs.sequences is [compact_prompt | generated]).
                pl = int(prompt_len_tokens) if prompt_len_tokens is not None else (int(prompt_len) if prompt_len is not None else seq_len)
                pl = min(pl, seq_len)
                text_outputs = []
                for b in range(batch_size):
                    gen_ids = cont[b, pl:].tolist()
                    safe_ids = [_safe_id(i) for i in gen_ids]
                    text_outputs.append(self.tokenizer.decode(safe_ids, skip_special_tokens=True).strip())
                # Per-sample debug: use logger.debug(); enable with setLevel(logging.DEBUG) on this logger
                _seq_len = cont.shape[1] if cont.dim() > 1 else 0
                _gen_len = int(gen_len) if gen_len is not None else max(0, _seq_len - pl)
                _resp0 = text_outputs[0] if text_outputs else ""
                logger.debug(
                    "[Exp1] prompt_len=%s gen_len=%s vision_tokens_K=%s seq_len=%s resp_len=%s resp_preview=%s",
                    prompt_len, gen_len, vision_tokens_K, _seq_len, len(_resp0), repr(_resp0[:120]),
                )
                _gen_len_safe = max(0, int(gen_len)) if gen_len is not None else max(0, _seq_len - pl)
                # FLOPs: use embed-space prompt_len (LLM input seq length). For Exp1 = text + K vision tokens.
                _pl = int(prompt_len) if prompt_len is not None else _seq_len
                results = self.analyzer.analyze_generate_task(
                    prompt_len=_pl,
                    gen_len=_gen_len_safe,
                    num_heads=num_heads,
                    batchsize=1,
                    w_bit=16,
                    a_bit=16,
                    kv_bit=16,
                    use_flashattention=False,
                )
                # Expose pruned token count and lengths so test/sweep output and samples JSONL show them.
                results["prompt_len"] = _pl
                results["gen_len"] = _gen_len_safe
                if vision_tokens_K is not None:
                    results["vision_tokens_K"] = vision_tokens_K
                # Per-sample FLOPs debug: use logger.debug(); enable with setLevel(logging.DEBUG) on this logger
                vision_placeholder_N = getattr(self.model, "_last_vision_placeholder_N", None)
                logger.debug(
                    "[Exp1 FLOPs debug] N=%s K=%s prompt_len=%s prefill_flops=%s",
                    vision_placeholder_N, vision_tokens_K, _pl, results.get("prefill_flops", 0),
                )

            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            results.update({'resp': text_outputs})
            res.extend([results])
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
