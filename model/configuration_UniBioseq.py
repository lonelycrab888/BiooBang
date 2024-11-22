# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
""" UniBioseqLM model configuration"""

from typing import Optional
from dataclasses import asdict, dataclass
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class UBSLMConfig(PretrainedConfig):
    r"""
    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the ESM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ESMModel`].
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the ESM code use this instead of the attention mask.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.
    ```"""

    model_type = "UniBioseqLM"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vocab_size=41,
        mask_token_id=40,
        pad_token_id=1,
        prompt_token_id=36,
        eos_token_id =2,

        hidden_size=1280,
        num_hidden_layers=33,
        num_attention_heads=20,
        intermediate_size=5120,

        token_dropout=False,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,

        max_position_embeddings=3079,
        initializer_range=0.02,
        layer_norm_eps=1e-12,

        need_weights = False, 
        output_attentions = False,
        output_hidden_states = False,

        add_zero_attn = False,
        bias = True,
        
        gradient_checkpointing=True,

        is_decoder = False,
        is_prefix_decoder = False,
        use_cache = False,
        
        mode = None,#"structure_emb_train"/"RAFold"
        problem_type = "regression",
        pooling_type = "MEAN",
        num_labels = 1,


        SE_loss_task = None,

        **kwargs,
    ):
        """
        token_dropout is True in ESM2, we choose False
        train prefix decoder: is_decoder=False,  use_cache=False,  is_prefix_decoder=False
        generate with prefix decoder: is_decoder=True,  use_cache=True,  is_prefix_decoder=True
        """
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs)


        self.prompt_token_id = prompt_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_cache = use_cache
        self.token_dropout = token_dropout
        self.add_zero_attn = add_zero_attn
        self.need_weights = need_weights
        self.is_decoder = is_decoder
        self.is_prefix_decoder = is_prefix_decoder
        self.bias = bias
        self.mode = mode
        self.problem_type = problem_type
        self.pooling_type = pooling_type
        self.num_labels = num_labels
        self.SE_loss_task = SE_loss_task
        
        
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        return output









    

def get_default_vocab_list():
    return (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        'a',
        'u',
        'c', 
        'g', 
        'n',
        '+',
        '<null_1>', 
        '<null_2>', 
        '<null_3>',
        "<mask>",
    )
