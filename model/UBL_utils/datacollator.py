import math
import random
import torch
from typing import Any, Optional, Tuple, List, Dict, Union
from collections.abc import Mapping
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm_probability=0.15):
        super().__init__(tokenizer, pad_to_multiple_of=None, mlm = True, mlm_probability = mlm_probability)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        replace_prompt_words = torch.randint(36, labels.shape, dtype=torch.long)
        
        not_replace = self.tokenizer.prompt_token_id
        if torch.any(random_words == not_replace):
            indices_to_replace = (random_words == not_replace)
            random_words[indices_to_replace] = replace_prompt_words[indices_to_replace]
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class PretrainDataCollatorForCDS(CustomDataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length=3079, mlm_probability=0.15):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.cls_id = tokenizer.cls_token_id
        self.prompt_id = tokenizer.prompt_token_id
        self.eos_id = tokenizer.eos_token_id
        self.max_length = max_length
    
    def __call__(self, features):
        #features = [item for pair in zip(features, features) for item in pair]
        #i = 0
        for feature in features:
            protein_ids_list = feature['protein_ids']
            cds_ids_list = feature['cds_ids']
            protein_ids0 = protein_ids_list[0]
            cds_ids0 = cds_ids_list[0]
            if len(protein_ids_list)>1:
                protein_ids0 = protein_ids_list[0]
                cds_ids0 = cds_ids_list[0]
            
                random_number = random.random()
                if random_number<=0.5:
                    protein_ids = protein_ids0
                    cds_ids = cds_ids0
                else:
                    random_choose = random.randint(0, len(protein_ids_list)-1)
                    protein_ids = protein_ids_list[random_choose]
                    cds_ids = cds_ids_list[random_choose]
            else:
                protein_ids = protein_ids0
                cds_ids = cds_ids0

            idx = feature['idx']
            len_total = 4 + len(protein_ids) + len(cds_ids)
            if len_total > self.max_length:
                splite_protein_length = math.floor((self.max_length - 4 - 3)/4)  #减去4个special token、及cds终止密码子长度
                splite_cds_length = splite_protein_length * 3

                protein_start_index = random.randint(0, len(protein_ids) - splite_protein_length)

                cds_start_index = 3*(protein_start_index)
                protein_new_ids = protein_ids[protein_start_index : protein_start_index + splite_protein_length] 
                cds_new_ids = cds_ids[cds_start_index : cds_start_index + splite_cds_length]
                if protein_start_index == len(protein_ids) - splite_protein_length:
                    cds_new_ids = cds_new_ids + cds_ids[-3:]

                if idx%2 == 0:
                    feature['input_ids'] = torch.tensor([self.cls_id] + protein_new_ids + [self.eos_id, self.prompt_id] + cds_new_ids +[self.eos_id])
                else:
                    feature['input_ids'] = torch.tensor([self.cls_id] + cds_new_ids + [self.eos_id, self.prompt_id] + protein_new_ids +[self.eos_id])
                
            else:
                if idx%2 == 0:
                    feature['input_ids'] = torch.tensor([self.cls_id] + protein_ids + [self.eos_id, self.prompt_id] + cds_ids +[self.eos_id])
                else:
                    feature['input_ids'] = torch.tensor([self.cls_id] + cds_ids + [self.eos_id, self.prompt_id] + protein_ids +[self.eos_id])
            
            
            del feature['protein_ids']
            del feature['cds_ids']
            del feature['idx']
            
        batch = super().__call__(features)
        return batch




class PretrainDataCollatorForCDS_test(CustomDataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length=3079, mlm_probability=0.15):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.cls_id = tokenizer.cls_token_id
        self.prompt_id = tokenizer.prompt_token_id
        self.eos_id = tokenizer.eos_token_id
        self.max_length = max_length
    
    def __call__(self, features):
        #features = [item for pair in zip(features, features) for item in pair]
        #i = 0
        for feature in features:
            protein_ids = feature['Protein']
            cds_ids = feature['CDS']
            idx = feature['idx']
            len_total = 4 + len(protein_ids) + len(cds_ids)
            if len_total > self.max_length:
                splite_protein_length = math.floor((self.max_length - 4 - 3)/4)  #减去4个special token、及cds终止密码子长度
                splite_cds_length = splite_protein_length * 3
                protein_start_index = random.randint(0, len(protein_ids) - splite_protein_length - 1)
                cds_start_index = 3*(protein_start_index)
                protein_new_ids = protein_ids[protein_start_index : protein_start_index + splite_protein_length] 
                cds_new_ids = cds_ids[cds_start_index : cds_start_index + splite_cds_length]+cds_ids[-3:]
                if idx%2 == 0:
                    feature['input_ids'] = torch.tensor([self.cls_id] + protein_new_ids + [self.eos_id, self.prompt_id] + cds_new_ids +[self.eos_id])
                else:
                    feature['input_ids'] = torch.tensor([self.cls_id] + cds_new_ids + [self.eos_id, self.prompt_id] + protein_new_ids +[self.eos_id])
                
            else:
                if idx%2 == 0:
                    feature['input_ids'] = torch.tensor([self.cls_id] + protein_ids + [self.eos_id, self.prompt_id] + cds_ids +[self.eos_id])
                else:
                    feature['input_ids'] = torch.tensor([self.cls_id] + cds_ids + [self.eos_id, self.prompt_id] + protein_ids +[self.eos_id])
                
            del feature['Protein']
            del feature['CDS']
            del feature['idx']
        batch = super().__call__(features)
        return batch



class PretrainDataCollatorFormRNA(CustomDataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length=3079, mlm_probability=0.15):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.cls_id = tokenizer.cls_token_id
        self.prompt_id = tokenizer.prompt_token_id
        self.eos_id = tokenizer.eos_token_id
        self.max_length = max_length
    
    def __call__(self, features):
        #features = [item for pair in zip(features, features) for item in pair]
        #i = 0
        for feature in features:
            protein_ids = feature['Protein']
            utr5_ids = feature['5UTR']
            cds_ids = feature['CDS']
            utr3_ids = feature['3UTR']
            idx = feature['idx']
            len_total = 4 + len(protein_ids) + len(utr5_ids) + len(utr3_ids) + len(cds_ids)
            if len_total > 3079:
                splite_protein_length = math.floor((3079 - 4- len(utr5_ids) - len(utr3_ids) - 3)/4)  #减去4个special token、utr长度、及cds终止密码子长度
                splite_cds_length = splite_protein_length*3
                protein_start_index = random.randint(0, len(protein_ids) - splite_protein_length - 1)
                cds_start_index = 3*(protein_start_index)
                protein_new_ids = protein_ids[protein_start_index : protein_start_index + splite_protein_length] 
                cds_new_ids = cds_ids[cds_start_index : cds_start_index + splite_cds_length]+cds_ids[-3:]
                #feature['input_ids'] = torch.tensor([self.cls_id] + protein_new_ids + [self.eos_id, self.prompt_id] + utr5_ids + cds_new_ids +utr3_ids +[self.eos_id])
                if idx%2 == 0:
                    feature['input_ids'] = torch.tensor([self.cls_id] + protein_new_ids + [self.eos_id, self.prompt_id] + utr5_ids + cds_new_ids +utr3_ids +[self.eos_id])
                else:
                    feature['input_ids'] = torch.tensor([self.cls_id] + utr5_ids + cds_new_ids +utr3_ids + [self.eos_id, self.prompt_id] + protein_new_ids +[self.eos_id])
                
            else:
                #feature['input_ids'] = torch.tensor([self.cls_id] + protein_ids + [self.eos_id, self.prompt_id] + utr5_ids + cds_ids +utr3_ids +[self.eos_id])
                
                if idx%2 == 0:
                    feature['input_ids'] = torch.tensor([self.cls_id] + protein_ids + [self.eos_id, self.prompt_id] + utr5_ids + cds_ids +utr3_ids +[self.eos_id])
                else:
                    feature['input_ids'] = torch.tensor([self.cls_id] + utr5_ids + cds_ids +utr3_ids + [self.eos_id, self.prompt_id] + protein_ids +[self.eos_id])
                
            del feature['Protein']
            del feature['5UTR']
            del feature['CDS']
            del feature['3UTR']
            del feature['idx']
            #i = i + 1
        batch = super().__call__(features)
        #print(batch)
        return batch
    

    



class DataCollatorForCDSGeneration(CustomDataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_length=3079, mlm_probability=0.2):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.cls_id = tokenizer.cls_token_id
        self.prompt_id = tokenizer.prompt_token_id
        self.eos_id = tokenizer.eos_token_id
        self.max_length = max_length
    
    def __call__(self, features):
        batch = super().__call__(features)
        return batch

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded
def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

class DataCollatorForCDSDecoder(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, pad_to_multiple_of=None, mlm = False)
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        x = batch["input_ids"].clone()
        for i in range(x.shape[0]):
            indices = (x[i] == 36).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                first_index = indices[0]
                x[i, :first_index+1] = -100
        x = x[:, 1:]
        new_col = torch.full((x.shape[0], 1), 2)
        labels = torch.cat((x, new_col), dim=1)
        for i in range(labels.shape[0]):
            if self.tokenizer.pad_token_id in labels[i]:
                labels[i, -1] = self.tokenizer.pad_token_id
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels
        
        return batch




class DPODataCollatorWithPaddingForCDS:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 1
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch

"""
class fullprotein_stucture_data_collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, examples):
        input_ids1 = [example['input_ids'] for example in examples]
        padded_input_ids1 = DataCollatorWithPadding(self.tokenizer, padding='longest')(input_ids1)
        input_ids2 = [example['input_ids2'] for example in examples]
        padded_input_ids2 = DataCollatorWithPadding(self.tokenizer, padding='longest')(input_ids2)
        return {
            "input_ids": padded_input_ids1['input_ids'],
            "input_ids2": padded_input_ids2['input_ids'],
            "position_ids":None,
            "position_ids2": None,
            "labels": example['labels']
        }
"""
class fullprotein_stucture_data_collator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_1 = []
        features_2 = []
        for i in features:
            features_1.append({'input_ids': i['input_ids'],'label': i['labels']})
            features_2.append({'input_ids': i['input_ids2']})
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features_1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_2 = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features_2,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch['input_ids2'] = batch_2['input_ids']
        batch['attention_mask2'] = batch_2['attention_mask']
        if "label" in batch:
            batch["labels"] = batch["label"].to(dtype = torch.float16)
            del batch["label"]
        return batch



