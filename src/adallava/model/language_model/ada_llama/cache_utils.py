from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers.cache_utils import DynamicCache

class DynamicCacheWithExecutionPlan(DynamicCache):
    """
    A cache that can handle execution plan

    It stores the Key and Value states as a dict of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    It also stores execution plan, which is a dict of integers, indicating the states of the decoder layers.
    """

    def __init__(self, n_layers) -> None:
        self.n_layers = n_layers
        self.key_cache: Dict[torch.Tensor] = {}
        self.value_cache: Dict[torch.Tensor] = {}
        self.execution_plan: Dict[int] = {}
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx in self.key_cache:
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache for layer {layer_idx} is missing or dropped, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            if layer_idx in self.key_cache:
                yield (self.key_cache[layer_idx], self.value_cache[layer_idx])
            else:
                continue

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return self.n_layers
    
    def update_execution_plan(self, drop_states, layer_idx):
        if layer_idx not in self.execution_plan:
            self.execution_plan[layer_idx] = drop_states

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        # TODO:Review
        for layer_idx in self.key_cache:
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        # TODO:Review
        legacy_cache = ()
        for layer_idx in range(len(self)):
            if layer_idx in self.key_cache:
                legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.execution_plan[layer_idx]),)
            else:
                legacy_cache += ((None, None, self.execution_plan[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, n_layers: int = 24) -> "DynamicCacheWithExecutionPlan":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        # TODO:Review
        cache = cls(n_layers)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                layer_entry = past_key_values[layer_idx]
                # Support both (key, value, drop_states) and (key, value) legacy formats.
                if len(layer_entry) >= 3:
                    key_states, value_states, drop_states = layer_entry[0], layer_entry[1], layer_entry[2]
                else:
                    key_states, value_states, drop_states = layer_entry[0], layer_entry[1], None
                if key_states is not None:
                    cache.update(key_states, value_states, layer_idx)
                cache.update_execution_plan(drop_states, layer_idx)
        return cache
    
    def get_execution_plan(self) -> List:
        # execution_plan can be a dict (from from_legacy_cache) or a list (assigned in model forward).
        # Use .get for dict so missing keys (e.g. legacy cache without execution_plan) yield None → full run.
        n = len(self)
        if isinstance(self.execution_plan, dict):
            return [self.execution_plan.get(layer_idx) for layer_idx in range(n)]
        return [self.execution_plan[layer_idx] for layer_idx in range(n)]
    
    @classmethod
    def get_execution_plan_from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, n_layers: int = 24) -> List:
        cache = cls.from_legacy_cache(past_key_values, n_layers)
        return cache.get_execution_plan()