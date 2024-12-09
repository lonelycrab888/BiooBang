import math
import warnings
from typing import Optional, Tuple, List, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import LayerNorm as UBSLMLayerNorm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SmoothL1Loss

from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast, 
    SequenceClassifierOutputWithPast, 
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
import transformers.models.convbert as c_bert
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

from .configuration_UniBioseq import UBSLMConfig
from .UBL_utils import  DynamicCache

logger = logging.get_logger(__name__)


def get_special_attn_mask(tokens, nheads, pad_id, prompt_id, is_prefix_decoder = False, is_clm = False, attn_len = None, device = 'cuda'):
    seq_num = tokens.shape[0]
    seq_len = tokens.shape[1]
    
    prompt = torch.where(tokens[0]==prompt_id)[0]
    if len(prompt)==0:
        if is_prefix_decoder and not is_clm:
            attn_mask = torch.zeros(attn_len , attn_len, device=tokens.device)
            temp_mask = torch.ones(attn_len, attn_len, dtype=torch.bool, device = tokens.device).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_mask = attn_mask[-seq_len:,:]
            attn_mask = attn_mask.unsqueeze(0).expand(nheads, -1, -1)
            attn_mask = attn_mask.unsqueeze(0).expand(seq_num, - 1, -1, -1)
            return attn_mask
        elif is_prefix_decoder and is_clm:
            attn_mask = torch.zeros(attn_len , attn_len, device=tokens.device)
            temp_mask = torch.ones(attn_len, attn_len, dtype=torch.bool, device = tokens.device).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_mask = attn_mask[-seq_len:,:]
            attn_mask = attn_mask.unsqueeze(0).expand(nheads, -1, -1)
            attn_mask = attn_mask.unsqueeze(0).expand(seq_num, - 1, -1, -1)
            return attn_mask

        else:
            attn_mask = torch.zeros(seq_num, seq_len, seq_len).to(device)
    else:
        attn_mask = torch.empty(0,seq_len,seq_len).to(device)
        for i in tokens:
            prompt_idex = (torch.where(i==prompt_id)[0][0].item())-1
            attn_mask_mlm = torch.cat((torch.zeros(prompt_idex+1,prompt_idex+1).to(device),-10000*torch.ones(prompt_idex+1, seq_len-prompt_idex-1).to(device)),dim=1)
            attn_mask_clm = torch.cat((torch.zeros(seq_len-prompt_idex-1,prompt_idex+1).to(device), torch.triu(-10000*torch.ones(seq_len-prompt_idex-1,seq_len-prompt_idex-1).to(device), diagonal=1)),dim=1)
            attn_mask_one = torch.cat((attn_mask_mlm,attn_mask_clm),dim=0)
            attn_mask = torch.cat((attn_mask, torch.unsqueeze(attn_mask_one,dim=0)),dim=0).to(device)
    mask_result = torch.empty(0,seq_len,seq_len).to(device)
    for i in range(seq_num):
        seq = tokens[i]
        mask = attn_mask[i]
        pad_idex = torch.where(seq==pad_id)[0]
        if len(pad_idex)!=0:
            pad_idex = pad_idex[0].item()
            attn_mask_lm = torch.split(mask, pad_idex, dim = 0)[0]
            attn_mask_lm = torch.split(attn_mask_lm, pad_idex, dim = 1)[0]
            attn_mask_lm = torch.cat((attn_mask_lm, -10000*torch.ones(pad_idex, seq_len-pad_idex).to(device)), dim=1)
            attn_mask_lm = torch.cat((attn_mask_lm, -10000*torch.ones(seq_len-pad_idex, seq_len).to(device)), dim=0)
        else:
            attn_mask_lm = mask
        for _ in range(nheads):
            mask_result = torch.cat((mask_result,torch.unsqueeze(attn_mask_lm,dim=0)),dim=0)
    #print(mask_result)
    return mask_result.to(device)


    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->UniBioseqLM
class UBSLMRotaryEmbedding(nn.Module):
    def __init__(self, dim,  max_position_embeddings=3076, base=10000, device=None):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)
        
    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )
    
# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_UBSLM(q, k, cos, sin, position_ids):
    if position_ids is None:
        seq_len = k.shape[-2]
        position_ids = torch.arange(seq_len).unsqueeze(0)
    #print(cos)
    cos = cos[position_ids]
    sin = sin[position_ids]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_unequ_pos(q, k, cos, sin, position_ids):
    if position_ids is None:
        seq_len = k.shape[-2]
        position_ids = torch.arange(seq_len).unsqueeze(0)
    cos = cos[position_ids]
    sin = sin[position_ids]
    bsz, nhead =cos.size(0), q.size(0) // cos.size(0)
    q = q.view(bsz, nhead, q.size(1), q.size(2))
    k = k.view(bsz, nhead, k.size(1), k.size(2))
    cos = cos.unsqueeze(1).expand(-1, nhead, -1, -1) 
    sin = sin.unsqueeze(1).expand(-1, nhead, -1, -1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.reshape(bsz*nhead, q_embed.size(2), q_embed.size(3))
    k_embed = k_embed.reshape(bsz*nhead, k_embed.size(2), k_embed.size(3))
    return q_embed, k_embed

def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

#Copied from facebook/esm
def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized



def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, scale=None) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weights_float = attn_weight
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weights_float

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class MultiheadAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = self.num_heads * self.head_dim
        self.layer_idx = layer_idx
        self.hidden_dropout = config.hidden_dropout_prob
        self.attn_dropout = config.attention_probs_dropout_prob
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.bias)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.bias)

        self.bias_k = self.bias_v = None
        self.add_zero_attn = config.add_zero_attn
        
        self.is_prefix_decoder = config.is_prefix_decoder
        self.rot_emb = UBSLMRotaryEmbedding(dim=self.head_dim, max_position_embeddings=config.max_position_embeddings)
        self.config = config
        self.reset_parameters()
        

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        attn_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        need_weights:bool = False,
        past_key_value: Optional[Cache] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        #Input shape: Time x Batch x Channel
        if output_attentions:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        assert k is not None
        src_len = k.shape[-2]
        if past_key_value is not None:
            if not self.is_prefix_decoder:
                src_len += past_key_value.get_usable_length(src_len, self.layer_idx)
            else:
                if src_len != 2:
                    src_len += past_key_value.get_usable_length(src_len, self.layer_idx)
                else:
                    src_len += past_key_value.get_usable_length(src_len, self.layer_idx)-1
                
        #cos, sin, q, k = self.rot_emb(q, k)
        if self.config.mode != "structure_emb_train":
            cos, sin = self.rot_emb(v, seq_len=src_len)
            q, k = apply_rotary_pos_emb_UBSLM(q, k, cos, sin, position_ids)
        else:
            cos, sin = self.rot_emb(v, 8650)
            q, k = apply_rotary_pos_emb_unequ_pos(q, k, cos, sin, position_ids)
        
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, tgt_len, self.head_dim)
        v = v.view(bsz, self.num_heads, tgt_len, self.head_dim)

        if past_key_value is not None :
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            k, v = past_key_value.update(k, v, self.layer_idx, self.is_prefix_decoder, self.config.is_clm, cache_kwargs)

        k = repeat_kv(k, 1)
        v = repeat_kv(v, 1)
        """
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        """
        if need_weights:
            attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, tgt_len).to(q.dtype)
            attn, attn_weights_float = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p = self.attn_dropout if self.training else 0.0, scale = 1)
            attn = attn.reshape(bsz*self.num_heads, tgt_len, self.head_dim)
        elif attn_mask is not None:
            attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, src_len).to(q.dtype)
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if attn_mask.device != torch.device('cpu'):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p = self.attn_dropout if self.training else 0.0, scale = 1)
                    attn = attn.reshape(bsz*self.num_heads, tgt_len, self.head_dim)
            else:
                attn, _ = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p = self.attn_dropout if self.training else 0.0, scale = 1)
                attn = attn.reshape(bsz*self.num_heads, tgt_len, self.head_dim)
        else:#for decoder_only model
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p = self.attn_dropout if self.training else 0.0,  is_causal=True,  scale = 1)
                attn = attn.reshape(bsz*self.num_heads, tgt_len, self.head_dim)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        
        attn = self.out_proj(attn)

        attn = F.dropout(
            attn,
            p = self.hidden_dropout,
            training=self.training,
        )

        attn_weights: Optional[Tensor] = None
        
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).type_as(attn).transpose(1, 0)
            if not output_attentions:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        
        return attn, attn_weights, past_key_value




class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        config,
        layer_idx = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dropout = config.hidden_dropout_prob
        self._init_submodules(config)

    def _init_submodules(self, config):
        BertLayerNorm = UBSLMLayerNorm 

        self.self_attn = MultiheadAttention(
            config,
            layer_idx= self.layer_idx
        )
        self.self_attn_layer_norm = BertLayerNorm(config.hidden_size)

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.final_layer_norm = BertLayerNorm(config.hidden_size)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        need_weights: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn, present_key_value= self.self_attn(
            query=hidden_states,
            need_weights=need_weights,
            attn_mask=attention_mask,
            output_attentions=output_attentions,
            past_key_value = past_key_value,
            position_ids = position_ids
        )
        hidden_states = residual + hidden_states

        #Fully connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p = self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class UniBioseqPreTrainedModel(PreTrainedModel):
    config_class = UBSLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerLayer", "EsmFoldTriangularSelfAttentionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)






class UniBioseqModel(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.padding_idx = config.pad_token_id
        self.mask_idx = config.mask_token_id
        self.prompt_idx = config.prompt_token_id

        self.attention_heads = config.num_attention_heads #20
        self.num_layers = config.num_hidden_layers #33

        self.token_dropout = config.token_dropout
        self._init_submodules(config)

        self.gradient_checkpointing = False
         # Initialize weights and apply final processing
        self.post_init()


    def _init_submodules(self, config):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx,)
        self.layers = nn.ModuleList(
            [TransformerLayer(config,layer_idx = layer_idx)for layer_idx in range(self.num_layers)]
            )
        
        self.emb_layer_norm_after = UBSLMLayerNorm(config.hidden_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            need_weights: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_decoder: Optional[bool] = None,
            )-> Union[Tuple, BaseModelOutputWithPast]:
        input_ids = input_ids.to(dtype = torch.long)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if is_decoder:
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            assert input_ids.ndim == 2
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        
        padding_mask = input_ids.eq(self.padding_idx)  # B, T
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        if self.token_dropout:
            hidden_states.masked_fill_((input_ids == self.mask_idx).unsqueeze(-1), 0.0)  #paddign_id's embdedding to 0
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)  #useful token length
            mask_ratio_observed = (input_ids == self.mask_idx).sum(-1).to(hidden_states.dtype) / src_lengths #mask_ration
            hidden_states = hidden_states * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None] 
        if padding_mask is not None:
            hidden_states = hidden_states * (1 - padding_mask.unsqueeze(-1).type_as(hidden_states))

            
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values, self.config.is_prefix_decoder, self.config.is_clm)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if is_decoder:
            if self.config.is_prefix_decoder and not self.config.is_clm:
                attn_len = position_ids[-1][-1]+1
                attention_mask = get_special_attn_mask(input_ids, 
                                                    self.attention_heads, 
                                                    self.padding_idx, 
                                                    self.prompt_idx, 
                                                    self.config.is_prefix_decoder, 
                                                    self.config.is_clm,
                                                    attn_len, 
                                                    device = input_ids.device)
            elif self.config.is_prefix_decoder and self.config.is_clm:
                attn_len = position_ids[-1]+1
                attention_mask = get_special_attn_mask(input_ids, 
                                                   self.attention_heads, 
                                                   self.padding_idx, 
                                                   self.prompt_idx, 
                                                   self.config.is_prefix_decoder,
                                                   self.config.is_clm,
                                                   attn_len,
                                                   device = input_ids.device)
            else :
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    hidden_states,
                    past_key_values_length,
                )
        else:
            attention_mask = get_special_attn_mask(input_ids, 
                                                   self.attention_heads, 
                                                   self.padding_idx, 
                                                   self.prompt_idx, 
                                                   device = input_ids.device)
        #print(attention_mask)


        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
         

        # (B, T, E) => (T, B, E)
        hidden_states = hidden_states.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(0, 1),)


            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    need_weights,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask = attention_mask,
                    position_ids =position_ids,
                    past_key_value = past_key_values,
                    need_weights = need_weights,
                    output_attentions=output_attentions,
                    use_cache = use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1].transpose(1,0),)

        hidden_states = self.emb_layer_norm_after(hidden_states)
        hidden_states = hidden_states.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = UBSLMLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))
        

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class UniBioseqForCausalLM(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = UniBioseqModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = RobertaLMHead(
            embed_dim=config.hidden_size,
            output_dim=config.vocab_size,
            weight=self.model.embed_tokens.weight,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        need_weights: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_decoder: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        is_decoder = is_decoder if is_decoder is not None else self.config.is_decoder
        need_weights =need_weights if need_weights is not None else self.config.need_weights
        is_prefix_decoder = self.config.is_prefix_decoder
        is_clm = self.config.is_clm
        if is_clm and position_ids is not None:
            position_ids = position_ids[0]
        elif is_clm and position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1]).to(device = input_ids.device)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None and is_decoder:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        elif labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if self.config.is_prefix_decoder and not self.config.is_clm:
            input_ids = torch.cat((input_ids,torch.ones(input_ids.shape[0],1, device = input_ids.device, dtype = input_ids.dtype)*self.config.mask_token_id), dim=1)
            seq_len = input_ids.shape[1]
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                if self.config.is_prefix_decoder and not self.config.is_clm:
                    input_ids = input_ids[:, past_length-1:]
                else:
                    input_ids = input_ids[:, past_length:]

                
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            #input_ids = torch.cat((input_ids,torch.tensor(self.config.mask_idx, device = input_ids.device)), dim=1)
            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            if not self.config.is_prefix_decoder or self.config.is_clm :
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]
            else:
                position_ids = torch.arange(seq_len).unsqueeze(0)
                if past_key_values:
                    position_ids = position_ids[:, -(input_ids.shape[1]) :]
                    

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        #print(attention_mask)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past






class UniBioseqForMaskedLM(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `UniBioseqForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.model = UniBioseqModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = RobertaLMHead(
            embed_dim=config.hidden_size,
            output_dim=config.vocab_size,
            weight=self.model.embed_tokens.weight,
        )
        self.post_init()

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            need_weights: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_decoder: Optional[bool] = None,
        ) -> Union[Tuple, MaskedLMOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        need_weights =need_weights if need_weights is not None else self.config.need_weights
        is_decoder = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class UBSLMForSequenceClassification(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = UniBioseqModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        need_weights: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_decoder: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = False
        output_hidden_states = False
        need_weights =False
        use_cache = False
        is_decoder = False
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class UniBioseqPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_type = config.pooling_type
        self.embed_dim = config.hidden_size
        self.eos_idx = config.eos_token_id
        
    def forward(self, feature, input_ids):
        if self.pooling_type == "MEAN":
            last_eos_positions = torch.zeros(input_ids.size(0),dtype=torch.long)
            for i, seq in enumerate(input_ids):
                eos_positions = (seq == self.eos_idx).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    last_eos_positions[i] = eos_positions[-1]

            hidden_state = torch.zeros(feature.size(0), self.embed_dim, device=feature.device)
            for i in range(feature.size(0)):
                state = feature[i,1:last_eos_positions[i],:]
                averaged_tensor = torch.mean(state, dim=0)
                #print(averaged_tensor)
                hidden_state[i] = averaged_tensor
            hidden_state = hidden_state.to(dtype=feature.dtype)
        elif self.pooling_type == "CLS":
            hidden_state = feature[:, 0, :]

        return hidden_state



#Copied from transformers.models.esm.modeling_esm.EsmClassificationHead with Esm->UniBioseq
"""
class UniBioseqClassificationHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
"""
"""
class UniBioseqClassificationHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels, bias=False)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x
"""

class UniBioseqClassificationHead(nn.Module):
    """Head for structure emb similarity tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(int(config.hidden_size/2), num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class UniBioseqMLPHead(nn.Module):
    """Head for structure emb similarity tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.intermediate_size, config.structure_hidden_size)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class UniBioseqForSequenceClassification_bidirectional(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = UniBioseqModel(config)
        self.pooling = UniBioseqPooler(config)
        self.score = UniBioseqClassificationHead(config, self.num_labels)

        self.post_init()

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            need_weights: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_decoder: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = False
        output_hidden_states = False
        need_weights =False
        use_cache = False
        is_decoder = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder,
        )
        sequence_output = outputs[0]
        sequence_pooling = self.pooling(sequence_output, input_ids)
        logits = self.score(sequence_pooling)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ConvBertHead(nn.Module):
    def __init__(self, config):
        """
        Base model that consists of ConvBert layer.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
            nlayers: Integer specifying the number of layers for the `ConvBert` model.
            kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
            pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
        """
        super().__init__()
        encoder_layers_Config = c_bert.ConvBertConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_convbert_heads,
            intermediate_size=config.convbert_intermediate_size,
            conv_kernel_size=config.conv_kernel_size,
            num_hidden_layers=config.num_hidden_convbert_layers,
            hidden_dropout_prob=config.convbert_hidden_dropout_prob,
        )

        self.transformer_encoder = nn.ModuleList(
            [c_bert.ConvBertLayer(encoder_layers_Config) for _ in range(1)]
        )
    
    def forward(self, x):
        for convbert_layer in self.transformer_encoder:
            x = convbert_layer(x)[0]
        return x


class UniBioseqForSequenceClassification_convbert(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = UniBioseqModel(config)
        self.convbert = ConvBertHead(config)
        self.pooling = UniBioseqPooler(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            need_weights: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_decoder: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = False
        output_hidden_states = False
        need_weights =False
        use_cache = False
        is_decoder = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder,
        )

        sequence_output = outputs[0]
        c_sequence_output = self.convbert(sequence_output)
        sequence_pooling = self.pooling(c_sequence_output, input_ids)
        logits = self.score(sequence_pooling)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class UniBioseqForEmbedding(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = UniBioseqModel(self.config)
        self.pooling = UniBioseqPooler(self.config)
        self.post_init()

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            position_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            need_weights: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_decoder: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = False
        output_hidden_states = False
        need_weights =False
        use_cache = False
        is_decoder = False

        output = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder,
        )
        hidden_states = output[0]
        logits = self.pooling(hidden_states, input_ids)

        loss = None
        dist = None

        if not return_dict:
            output = (logits, hidden_states,  None)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss = loss,
            logits = logits,
            hidden_states = hidden_states,
            attentions = None,
        )

class UniBioseqForTokenClassification(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = UniBioseqPreTrainedModel(config, add_pooling_layer=False)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()
    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            need_weights: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_decoder: Optional[bool] = None,
        ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = False
        output_hidden_states = False
        need_weights =False
        use_cache = False
        is_decoder = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder,
        )
        sequence_output = outputs[0]

        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs[3],
        )

class UniBioseqForDoubleregression_bidirectional(UniBioseqPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = UniBioseqModel(config)
        self.pooling = UniBioseqPooler(config)
        self.Regressionhead = UniBioseqClassificationHead(config, 1)
        self.Regressionhead_MFE = UniBioseqClassificationHead(config, 1)
        self.post_init()

    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            labels_MFE: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            need_weights: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_decoder: Optional[bool] = None,
        ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = False
        output_hidden_states = False
        need_weights =False
        use_cache = False
        is_decoder = False
        outputs = self.model(
            input_ids=input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            need_weights = need_weights,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            is_decoder = is_decoder,
        )
        sequence_output = outputs[0]
        sequence_pooling = self.pooling(sequence_output, input_ids)
        logits_1 = self.Regressionhead(sequence_pooling)
        logits_MFE = self.Regressionhead_MFE(sequence_pooling)
        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss1 = loss_fct(logits_1.squeeze(), labels.squeeze())
            loss2 = loss_fct(logits_MFE.squeeze(), labels_MFE.squeeze())
            loss = loss1 + loss2*loss1
            
        logits = torch.cat((logits_1, logits_MFE), dim=1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


