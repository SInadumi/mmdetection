# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except ImportError:
    AutoConfig = None
    AutoModel = None
    AutoTokenizer = None

from mmdet.registry import MODELS
from .utils import generate_masks_with_special_tokens_and_transfer_map


@MODELS.register_module()
class MDebertaModel(BaseModel):
    """mDeBERTa model for language embedding only encoder.

    Args:
        name (str, optional): name of the pretrained model from HuggingFace.
            Defaults to microsoft/mdeberta-v3-base.
        max_tokens (int, optional): maximum number of tokens to be used.
            Defaults to 256.
        pad_to_max (bool, optional): whether to pad the tokens to max_tokens.
            Defaults to True.
        use_sub_sentence_represent (bool, optional): whether to use sub
            sentence represent introduced in GroundingDINO. Defaults to False.
        special_tokens_list (list, optional): special tokens used to split
            subsentence. Required when use_sub_sentence_represent is True.
        num_layers_of_embedded (int, optional): number of layers of the
            embedded model. Defaults to 1.
        use_checkpoint (bool, optional): whether to use gradient checkpointing.
            Defaults to False.
    """

    def __init__(self,
                 name: str = 'microsoft/mdeberta-v3-base',
                 max_tokens: int = 256,
                 pad_to_max: bool = True,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          AutoHFEncoder(
                              name,
                              num_layers_of_embedded=num_layers_of_embedded,
                              use_checkpoint=use_checkpoint))]))

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens should not be None \
                    if use_sub_sentence_represent is True'

            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True).to(device)
        input_ids = tokenized.input_ids
        token_type_ids = tokenized.get('token_type_ids')

        if self.use_sub_sentence_represent:
            text_self_attention_masks, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens)
            attention_mask = tokenized.attention_mask.bool()
        else:
            text_self_attention_masks = None
            attention_mask = tokenized.attention_mask.bool()
            position_ids = None

        tokenizer_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        language_dict_features = self.language_backbone(tokenizer_input)
        if self.use_sub_sentence_represent:
            language_dict_features['position_ids'] = position_ids
            language_dict_features[
                'text_token_mask'] = tokenized.attention_mask.bool()
            language_dict_features['masks'] = text_self_attention_masks
        return language_dict_features


class AutoHFEncoder(nn.Module):
    """AutoModel encoder for language embedding."""

    def __init__(self,
                 name: str,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        if AutoConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = AutoConfig.from_pretrained(name)
        if hasattr(config, 'gradient_checkpointing'):
            config.gradient_checkpointing = use_checkpoint
        self.model = AutoModel.from_pretrained(
            name, config=config, use_safetensors=True)
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        inputs = {
            'input_ids': x['input_ids'],
            'attention_mask': mask,
        }
        if x.get('position_ids') is not None:
            inputs['position_ids'] = x['position_ids']
        if x.get('token_type_ids') is not None:
            inputs['token_type_ids'] = x['token_type_ids']

        outputs = self.model(**inputs, output_hidden_states=True)

        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:],
                               1).mean(1)
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
        return results
