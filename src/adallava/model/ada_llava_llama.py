#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

from .language_model.ada_llama.configuration_ada_llama import AdaLlamaConfig
from .language_model.ada_llama.modeling_ada_llama import AdaLlamaModel, AdaLlamaForCausalLM, CausalLMOutputWithPast
from .language_model.ada_llama.cache_utils import DynamicCacheWithExecutionPlan

from .scheduler.simple_scheduler import *
from .multimodal_encoder.prumerge_utils import prune_merge, prune_merge_plus


class AdaLlavaConfig(AdaLlamaConfig):
    model_type = "ada_llava_llama"

    def __init__(
        self,
        token_selecting="none",  # "none", "prumerge", "prumerge+", "adaptive"
        scheduler_type="L",  # "L", "H"
        scheduler_rank=8,
        vision_controller_budget_min=0.2,
        vision_controller_budget_max=1.0,
        vision_controller_tau=5.0,
        num_vision_patches=576,
        mm_hidden_size=None,  # vision encoder hidden size (e.g. 1024 for ViT-L); set from tower if None
        freeze_llm_scheduler=False,  # when True and token_selecting='adaptive': full LLM (no layer/head skip)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_selecting = token_selecting
        self.scheduler_type = scheduler_type
        self.scheduler_rank = scheduler_rank
        self.vision_controller_budget_min = vision_controller_budget_min
        self.vision_controller_budget_max = vision_controller_budget_max
        self.vision_controller_tau = vision_controller_tau
        self.num_vision_patches = num_vision_patches
        self.mm_hidden_size = mm_hidden_size
        self.freeze_llm_scheduler = freeze_llm_scheduler


class AdaLlavaLlamaModel(LlavaMetaModel, AdaLlamaModel):
    config_class = AdaLlavaConfig

    def __init__(self, config: AdaLlamaConfig):
        super(AdaLlavaLlamaModel, self).__init__(config)


class AdaLlavaLlamaForCausalLM(AdaLlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = AdaLlavaConfig
    # Checkpoint keys under this prefix are not applied during from_pretrained (vision_tower is
    # lazy-loaded later). They are loaded in builder.load_pretrained_model via
    # _load_vision_tower_from_checkpoint to avoid "Some weights were not used" warning.
    _keys_to_ignore_on_load_unexpected = [r"model\.vision_tower\..*"]

    def __init__(self, config):
        super(AdaLlamaForCausalLM, self).__init__(config)
        self.model = AdaLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.config.scheduler_type == "L":
            self.scheduler = SimpleScheduler_L(config)
        elif self.config.scheduler_type == "H":
            self.scheduler = SimpleScheduler_H(config)
        else:
            raise NotImplementedError

        if getattr(config, "token_selecting", "none") == "adaptive":
            from .multimodal_encoder.budget_embedding import BudgetEmbedding
            from .multimodal_encoder.vision_token_controller import VisionTokenController
            from .multimodal_encoder.vision_with_budget_token import VisionEncoderWithBudgetToken

            vision_dim = getattr(config, "mm_hidden_size", 1024)
            num_patches = getattr(config, "num_vision_patches", 576)
            tau = getattr(config, "vision_controller_tau", 5.0)
            self.budget_embedding = BudgetEmbedding(
                dim_out=vision_dim, num_freqs=128, hidden_dim=256
            )
            self.vision_token_controller = VisionTokenController(
                vision_dim=vision_dim,
                num_patches=num_patches,
                tau=tau,
            )
            self.vision_encoder_with_budget = VisionEncoderWithBudgetToken(
                budget_embedding_module=self.budget_embedding,
                vision_hidden_size=vision_dim,
            )
        else:
            self.budget_embedding = None
            self.vision_token_controller = None
            self.vision_encoder_with_budget = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def _get_token_budget(self, batch_size: int) -> torch.Tensor:
        """Sample token budget (ratio in [min, max]) for training. Used when token_selecting=='adaptive'.

        Note: returning per-sample budgets is fine because we do not change the module execution order
        (we only mask patch features to 0). This matches AdaLLaVA's training where latency is per-sample.
        """
        min_r = getattr(self.config, "vision_controller_budget_min", 0.2)
        max_r = getattr(self.config, "vision_controller_budget_max", 1.0)
        device = next(self.parameters()).device
        return torch.rand(batch_size, device=device, dtype=self.dtype) * (max_r - min_r) + min_r

    def encode_images(self, images, token_budget: Optional[torch.Tensor] = None, **kwargs):
        """Vision token scheduler (token_selecting='adaptive') only affects this path.

        - Training: output [B, N, C] with unselected positions zeroed; mm_projector and LLM
          see N positions (no sequence-length change; backward / straight-through work).
        - Evaluation: output [B, K, C] with only K selected tokens gathered; mm_projector and
          LLM see K positions so FLOPs truly decrease. LLaVA merge uses image_features.size(1)
          for the number of image token positions, so no change to prepare_inputs_labels_for_multimodal.
        """
        if type(images) is list or self.config.token_selecting == "none":
            return super().encode_images(images)
        elif self.config.token_selecting == "prumerge":
            vision_tower = self.get_model().get_vision_tower()
            image_features = prune_merge(vision_tower, images).to(self.dtype)
        elif self.config.token_selecting == "prumerge+":
            vision_tower = self.get_model().get_vision_tower()
            image_features = prune_merge_plus(vision_tower, images).to(self.dtype)
        elif self.config.token_selecting == "adaptive":
            vision_tower = self.get_model().get_vision_tower()
            images_in = images.to(device=self.device, dtype=self.dtype)
            batch_size = images_in.size(0)
            if token_budget is None:
                if self.training:
                    token_budget = self._get_token_budget(batch_size)
                else:
                    tb = getattr(self, "current_token_budget", None)
                    if tb is not None:
                        if isinstance(tb, (int, float)):
                            token_budget = torch.full(
                                (batch_size,), float(tb), device=self.device, dtype=self.dtype
                            )
                        else:
                            token_budget = tb if tb.numel() >= batch_size else tb.expand(batch_size)
                    else:
                        token_budget = torch.full(
                            (batch_size,), 1.0, device=self.device, dtype=self.dtype
                        )
            # Budget token appended to patch sequence -> ViT -> [B, N+2, C]; controller uses last pos -> Linear(C, N) -> logits.
            vision_output = self.vision_encoder_with_budget(
                images_in, token_budget, vision_tower=vision_tower
            ).to(self.dtype)
            selected, keep_mask = self.vision_token_controller(vision_output, token_budget)
            if not self.training:
                # Eval: gather only K kept tokens -> [B, K, C] so mm_projector and LLM see real sequence length K.
                # FLOPs then truly decrease (mm_projector and LLM over K, not N).
                patch_tokens = vision_output[:, 1:-1, :]  # [B, N, C]
                B, N, C = patch_tokens.shape
                k_per_batch = keep_mask.sum(dim=1)  # [B]
                max_k = k_per_batch.max().item()
                min_k = k_per_batch.min().item()
                if max_k == min_k:
                    # Same K for all (typical for fixed token_budget in eval sweep).
                    K = int(min_k)
                    idx_1d = keep_mask[0].nonzero(as_tuple=True)[0].long().to(device=patch_tokens.device)
                    indices = idx_1d.unsqueeze(0).expand(B, -1).unsqueeze(-1).expand(B, K, C)
                    gathered = torch.gather(patch_tokens, 1, indices)
                else:
                    # Different K per sample: gather per batch and pad to max_k (rare in eval).
                    gathered_list = []
                    for b in range(B):
                        idx = keep_mask[b].nonzero(as_tuple=True)[0]
                        g = patch_tokens[b : b + 1, idx, :]
                        if g.size(1) < max_k:
                            g = torch.nn.functional.pad(g, (0, 0, 0, max_k - g.size(1)), value=0.0)
                        gathered_list.append(g)
                    gathered = torch.cat(gathered_list, dim=0)
                image_features = self.get_model().mm_projector(gathered)
                # Expose K for eval logging (e.g. lmms_eval wrapper can log "vision_tokens_K" and FLOPs sanity-check).
                K_used = image_features.shape[1]
                object.__setattr__(self, "_last_vision_token_count", K_used)
                # Debug: log once per process (Exp1 empty-resp debugging)
                if not getattr(self, "_encode_images_eval_logged", False):
                    import logging
                    logging.getLogger(__name__).info(
                        f"[Exp1 encode_images] eval path: image_features.shape={image_features.shape} device={image_features.device}"
                    )
                    object.__setattr__(self, "_encode_images_eval_logged", True)
            else:
                # Training: keep [B, N, C] with zeros so backward and straight-through work; no sequence-length change.
                image_features = self.get_model().mm_projector(selected)
            return image_features
        else:
            raise NotImplementedError
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        latency: Optional[torch.FloatTensor] = None,
        latency_token_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        if self.training:
            # When training only the vision token scheduler (freeze_llm_scheduler + adaptive),
            # run full LLM: no latency token, no scheduler; execution_plan = full (no skip).
            if getattr(self.config, "freeze_llm_scheduler", False) and self.config.token_selecting == "adaptive":
                latency = None
                latency_token_position = None
            else:
                latency = self.scheduler.get_random_latency(inputs_embeds.size(0))
                latency_embeding = self.scheduler.latency_encoding(latency.to(self.device, dtype=self.dtype))
                inputs_embeds, position_ids, attention_mask, labels, latency_token_position = self.insert_latency_token(
                    inputs_embeds=inputs_embeds,
                    position_ids_=position_ids,
                    attention_mask_=attention_mask,
                    labels_=labels,
                    latency_embeding=latency_embeding,
                )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            latency=latency,
            latency_token_position=latency_token_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return outputs
    
    def insert_latency_token(self, inputs_embeds, latency_embeding, position_ids_, attention_mask_, labels_=None):
        if position_ids_ is None:
            position_ids = torch.arange(inputs_embeds.size(1)).unsqueeze(0).to(inputs_embeds.device)
        else:
            position_ids = position_ids_

        if attention_mask_ is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        else:
            attention_mask = attention_mask_ 

        if labels_ is None:
            labels = torch.full(inputs_embeds.shape[:2], IGNORE_INDEX, device=inputs_embeds.device)
        else:
            labels = labels_

        new_labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        inserting_position = [(label == IGNORE_INDEX).nonzero(as_tuple=True)[0][-1].item() + 1 for label in new_labels]
        
        new_inputs_embeds = []
        new_position_ids = []
        new_attention_mask = []
        new_labels = []

        for batch_idx in range(inputs_embeds.size(0)):
            new_inputs_embeds.append(
                torch.cat([inputs_embeds[batch_idx][:inserting_position[batch_idx]], 
                           latency_embeding[batch_idx:batch_idx+1], 
                           inputs_embeds[batch_idx][inserting_position[batch_idx]:]], 0)
                )
            
            new_attn_b = torch.cat([attention_mask[batch_idx][:inserting_position[batch_idx]], 
                           torch.ones((1,), dtype=attention_mask.dtype, device=attention_mask.device), 
                           attention_mask[batch_idx][inserting_position[batch_idx]:]], 0)
            new_attention_mask.append(new_attn_b)

            cur_position_ids = torch.arange(new_attn_b.size(-1), device=position_ids.device, dtype=position_ids.dtype)
            cur_position_ids[~new_attn_b] = 0
            new_position_ids.append(cur_position_ids)
            
            new_labels.append(
                torch.cat([labels[batch_idx][:inserting_position[batch_idx]], 
                        torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=labels.device), 
                        labels[batch_idx][inserting_position[batch_idx]:]], 0)
                )

            
        new_inputs_embeds = torch.stack(new_inputs_embeds, 0)
        new_position_ids = torch.stack(new_position_ids, 0)
        new_attention_mask = torch.stack(new_attention_mask, 0)
        new_labels = torch.stack(new_labels, 0)
        inserting_position = torch.tensor(inserting_position)

        if position_ids_ is None:
            new_position_ids = None
        
        if attention_mask_ is None:
            new_attention_mask = None

        if labels_ is None:
            new_labels = None
        
        return new_inputs_embeds, new_position_ids, new_attention_mask, new_labels, inserting_position

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        images: Optional[torch.FloatTensor],
        image_sizes: Optional[List[List[int]]] = None,
    ) -> Tuple[
        torch.LongTensor,
        Optional[torch.LongTensor],
        Optional[torch.Tensor],
        Optional[List[torch.FloatTensor]],
        torch.FloatTensor,
        Optional[torch.LongTensor],
    ]:
        """Override to support K image tokens in eval when token_selecting=='adaptive' (K < N).

        LLaVA's merge expects image_features.size(1) to match the number of IMAGE_TOKEN_INDEX
        in input_ids. When we use a token budget, encode_images returns [B, K, C] but the
        tokenized prompt has N placeholders. We shorten the sequence by keeping only K
        image placeholder positions so the parent merge gets K positions and K features.
        """
        if (
            getattr(self.config, "token_selecting", "none") != "adaptive"
            or self.training
            or images is None
        ):
            return super().prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=image_sizes
            )

        # Eval with adaptive: get K from encode_images, N from input_ids placeholder count.
        with torch.no_grad():
            image_features = self.encode_images(images, image_sizes=image_sizes)
        K = image_features.shape[1]
        # Number of image placeholder positions (assume one contiguous block per batch).
        is_img = input_ids == IMAGE_TOKEN_INDEX
        # Use first batch to get N; in VQA typically same N for all.
        first_row = is_img[0]
        if not first_row.any():
            return super().prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=image_sizes
            )
        img_indices = first_row.nonzero(as_tuple=True)[0]
        N = img_indices.numel()
        # Expose N for debug (wrapper can log N vs K; FLOPs use inputs_embeds.shape[1] in generate()).
        object.__setattr__(self, "_last_vision_placeholder_N", N)
        if K >= N:
            # Still force first return to be input_ids so generate() gets full sequence (parent may return None).
            out = super().prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=image_sizes
            )
            return (input_ids, out[1], out[2], out[3], out[4], out[5])

        # K < N: shorten sequence by removing (N - K) image placeholder positions per batch.
        B, L = input_ids.shape
        new_len = L - (N - K)
        new_input_ids = input_ids.new_zeros(B, new_len)
        new_position_ids = (
            position_ids.new_zeros(B, new_len).to(dtype=position_ids.dtype, device=position_ids.device)
            if position_ids is not None
            else None
        )
        new_attention_mask = (
            attention_mask.new_zeros(B, new_len, dtype=attention_mask.dtype, device=attention_mask.device)
            if attention_mask is not None
            else None
        )
        new_labels = (
            labels.new_full((B, new_len), IGNORE_INDEX, device=labels.device)
            if labels is not None
            else None
        )

        for b in range(B):
            row = is_img[b]
            if not row.any():
                new_input_ids[b] = input_ids[b, :new_len]
                if new_position_ids is not None and position_ids is not None:
                    new_position_ids[b] = position_ids[b, :new_len]
                if new_attention_mask is not None:
                    new_attention_mask[b] = attention_mask[b, :new_len]
                if new_labels is not None:
                    new_labels[b] = labels[b, :new_len]
                continue
            idx = row.nonzero(as_tuple=True)[0]
            start = idx[0].item()
            end = idx[-1].item() + 1
            # Keep: [ :start ], [ start : start+K ], [ end : ]
            new_input_ids[b] = torch.cat([
                input_ids[b, :start],
                input_ids[b, start : start + K],
                input_ids[b, end:],
            ], dim=0)
            if new_position_ids is not None and position_ids is not None:
                new_position_ids[b] = torch.cat([
                    position_ids[b, :start],
                    position_ids[b, start : start + K],
                    position_ids[b, end:],
                ], dim=0)
            if new_attention_mask is not None:
                new_attention_mask[b] = torch.cat([
                    attention_mask[b, :start],
                    attention_mask[b, start : start + K],
                    attention_mask[b, end:],
                ], dim=0)
            if new_labels is not None:
                new_labels[b] = torch.cat([
                    labels[b, :start],
                    labels[b, start : start + K],
                    labels[b, end:],
                ], dim=0)

        # Call parent merge; then force first return to be new_input_ids so generate() gets
        # shortened input_ids (parent may return None for first value → outputs.sequences only generated part).
        out = super().prepare_inputs_labels_for_multimodal(
            new_input_ids, new_position_ids, new_attention_mask, past_key_values, new_labels, images, image_sizes=image_sizes
        )
        # Ensure caller gets new_input_ids for building full sequence in generate().
        return (new_input_ids, out[1], out[2], out[3], out[4], out[5])

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        latency: Optional[torch.Tensor] = None,
        return_dict_in_generate: Optional[bool] = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if isinstance(latency, float):
            latency = torch.full((inputs_embeds.shape[0],), latency, dtype=self.dtype)

        # When only the vision token scheduler is used (freeze_llm_scheduler + adaptive), keep behavior aligned with
        # training: no latency token in sequence, and LLM runs all layers and heads (no layer/head scheduler).
        # Training does the same in forward(): latency=None, latency_token_position=None → LLM sets
        # execution_plan = [None] * num_hidden_layers (see modeling_ada_llama.py).
        is_adaptive_only = (
            getattr(self.config, "token_selecting", "none") == "adaptive"
            and getattr(self.config, "freeze_llm_scheduler", False)
        )
        if is_adaptive_only:
            latency_token_position = None
            latency_for_llm = None  # LLM then uses execution_plan = [None]*n → all layers/heads active
        else:
            latency_embeding = self.scheduler.latency_encoding(latency.to(self.device, dtype=self.dtype))
            inputs_embeds, position_ids, attention_mask, _, latency_token_position = self.insert_latency_token(
                inputs_embeds=inputs_embeds,
                position_ids_=position_ids,
                attention_mask_=attention_mask,
                labels_=None,
                latency_embeding=latency_embeding,
            )
            latency_for_llm = latency

        assert inputs_embeds.shape[0] == 1, "Batch size > 1 is not supported."

        # Align input_ids length with inputs_embeds when we inserted a latency token (no token id for that position).
        if inputs is not None and inputs.shape[1] + 1 == inputs_embeds.shape[1] and latency_token_position is not None:
            pos = int(latency_token_position[0].item())
            last_id = inputs[0, pos - 1].item() if pos > 0 else None
            if last_id is not None and 0 <= last_id < self.config.vocab_size:
                pad_id = last_id
            else:
                pad_id = getattr(self.config, "pad_token_id", None)
                if pad_id is None or pad_id == 0:
                    pad_id = getattr(self.config, "bos_token_id", 1)
                pad_id = max(0, min(int(pad_id), self.config.vocab_size - 1))
            inputs = torch.cat([
                inputs[:, :pos],
                torch.full((inputs.size(0), 1), pad_id, device=inputs.device, dtype=inputs.dtype),
                inputs[:, pos:],
            ], dim=1)

        outputs = super().generate(
            inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            latency=latency_for_llm,
            latency_token_position=latency_token_position,
            return_dict_in_generate=return_dict_in_generate,
            **kwargs
        )

        if return_dict_in_generate:
            # prompt_len (embed space): for FLOPs — actual sequence length the LLM sees (1+text+K).
            # prompt_len_tokens (token space): length of prompt in outputs.sequences. Parent builds
            # sequences from compact input_ids when available. When inputs is None (latency model path:
            # parent returns None from prepare_inputs_labels_for_multimodal), the parent's generate()
            # typically returns output.sequences containing only the generated tokens (no prompt ids),
            # so prompt_len_tokens = 0.
            prompt_len = int(inputs_embeds.shape[1])
            prompt_len_tokens = (
                int(inputs.shape[1]) if inputs is not None else 0
            )
            gen_len = outputs.sequences.shape[1] - prompt_len_tokens
            raw_plan = DynamicCacheWithExecutionPlan.get_execution_plan_from_legacy_cache(outputs.past_key_values, self.config.num_hidden_layers)
            num_heads = getattr(self.config, "num_attention_heads", 32)
            # None = full layer (no skip); tensor = scheduler output. For outputs we use int per layer (heads count).
            execution_plan = [
                num_heads if _ is None else (_.sum().item() // 2)
                for _ in raw_plan
            ]
            execution_plan = [-1 if _ == 0 else _ for _ in execution_plan]
            setattr(outputs, "prompt_len", prompt_len)
            setattr(outputs, "prompt_len_tokens", prompt_len_tokens)
            setattr(outputs, "gen_len", gen_len)
            setattr(outputs, "execution_plan", execution_plan)
            return outputs
        else:
            return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, latency=None, latency_token_position=None,
                                      **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if latency is not None:
            inputs['latency'] = latency
        if latency_token_position is not None:
            inputs['latency_token_position'] = latency_token_position
        return inputs
    
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, 
                                            is_encoder_decoder=False, 
                                            standardize_cache_format=False,
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs, 
            model_kwargs=model_kwargs, 
            is_encoder_decoder=is_encoder_decoder, 
            standardize_cache_format=standardize_cache_format
        )

        if "latency" in model_kwargs:
            model_kwargs["latency"] = None

        if "latency_token_position" in model_kwargs:
            model_kwargs["latency_token_position"] = None

        return model_kwargs

AutoConfig.register("ada_llava_llama", AdaLlavaConfig)
AutoModelForCausalLM.register(AdaLlavaConfig, AdaLlavaLlamaForCausalLM)