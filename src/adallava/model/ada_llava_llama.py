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
from llava.constants import IGNORE_INDEX

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


class AdaLlavaLlamaModel(LlavaMetaModel, AdaLlamaModel):
    config_class = AdaLlavaConfig

    def __init__(self, config: AdaLlamaConfig):
        super(AdaLlavaLlamaModel, self).__init__(config)


class AdaLlavaLlamaForCausalLM(AdaLlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = AdaLlavaConfig

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

            vision_dim = getattr(config, "mm_hidden_size", 1024)
            num_patches = getattr(config, "num_vision_patches", 576)
            tau = getattr(config, "vision_controller_tau", 5.0)
            # When mm_vision_select_feature == "cls_patch", vision_tower returns [B, N+1, C];
            # use_cls=True so controller keeps CLS and selects among N patches -> output [B, N+1, C].
            # When "patch" (default), vision_tower returns [B, N, C]; use_cls=False -> output [B, N, C].
            use_cls = getattr(config, "mm_vision_select_feature", "patch") == "cls_patch"
            self.budget_embedding = BudgetEmbedding(
                dim_out=vision_dim, num_freqs=128, hidden_dim=256
            )
            self.vision_token_controller = VisionTokenController(
                vision_dim=vision_dim,
                num_patches=num_patches,
                tau=tau,
                use_cls=use_cls,
            )
        else:
            self.budget_embedding = None
            self.vision_token_controller = None

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
            # IMPORTANT for ZeRO-3 stability:
            # Do NOT call the inner transformers CLIPVisionModel directly here because it returns a
            # ModelOutput object. DeepSpeed ZeRO-3 parameter offload hooks sometimes cannot detect
            # tensors embedded in custom output objects, leading to "Invalidate trace cache" /
            # "still have inflight params".
            #
            # Instead, call LLaVA's CLIPVisionTower wrapper, which returns a plain Tensor.
            # Shape: [B, N, C] if mm_vision_select_feature=='patch', [B, N+1, C] if 'cls_patch'.
            vision_output = vision_tower(images_in).to(self.dtype)

            batch_size = vision_output.size(0)
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
            budget_emb = self.budget_embedding(token_budget)
            selected, _ = self.vision_token_controller(
                vision_output, token_budget, budget_embedding=budget_emb
            )
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
            latency = self.scheduler.get_random_latency(inputs_embeds.size(0))
            latency_embeding = self.scheduler.latency_encoding(latency.to(self.device, dtype=self.dtype))
            inputs_embeds, position_ids, attention_mask, labels, latency_token_position = self.insert_latency_token(inputs_embeds=inputs_embeds, 
                                                                                                position_ids_=position_ids,
                                                                                                attention_mask_=attention_mask, 
                                                                                                labels_=labels, 
                                                                                                latency_embeding=latency_embeding)

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
            
            new_attention_mask.append(
                torch.cat([attention_mask[batch_idx][:inserting_position[batch_idx]], 
                           torch.ones((1,), dtype=attention_mask.dtype, device=attention_mask.device), 
                           attention_mask[batch_idx][inserting_position[batch_idx]:]], 0)
                )

            cur_position_ids = torch.arange(new_attention_mask[-1].size(-1))
            cur_position_ids[~new_attention_mask[-1]] = 0
            new_position_ids.append(cur_position_ids.to(position_ids.device, dtype=position_ids.dtype))
            
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

        latency_embeding = self.scheduler.latency_encoding(latency.to(self.device, dtype=self.dtype))
        inputs_embeds, position_ids, attention_mask, _, latency_token_position = self.insert_latency_token(inputs_embeds=inputs_embeds, 
                                                                                             position_ids_=position_ids,
                                                                                             attention_mask_=attention_mask, 
                                                                                             labels_=None, 
                                                                                             latency_embeding=latency_embeding)
        assert inputs_embeds.shape[0] == 1, "Batch size > 1 is not supported."

        outputs = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            latency=latency,
            latency_token_position=latency_token_position,
            return_dict_in_generate=return_dict_in_generate,
            **kwargs
        )
        
        if return_dict_in_generate:
            prompt_len = inputs_embeds.shape[1] + 1
            gen_len = outputs.sequences.shape[1] - 1
            execution_plan = DynamicCacheWithExecutionPlan.get_execution_plan_from_legacy_cache(outputs.past_key_values, self.config.num_hidden_layers)
            execution_plan = [_.sum().item()//2 for _ in execution_plan]
            execution_plan = [-1 if _ == 0 else _ for _ in execution_plan]
            setattr(outputs, 'prompt_len', prompt_len)
            setattr(outputs, 'gen_len', gen_len)
            setattr(outputs, 'execution_plan', execution_plan)
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