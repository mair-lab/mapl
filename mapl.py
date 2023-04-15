from pathlib import Path
from typing import List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel, AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from PIL import Image

from utils import accumulate_padding, torch_dtype, is_remote_url


logging.set_verbosity_error()
logger = logging.get_logger("transformers")


class LanguageDecoder(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.forward = self.model.forward
        self.generate = self.model.generate

    @property
    def model_id(self) -> str:
        return type(self.model).__name__.lower()

    @property
    def embed_dim(self) -> int:
        if 'gpt' in self.model_id:
            return self.model.config.n_embd
        elif 'opt' in self.model_id:
            return self.model.config.word_embed_proj_dim
        else:
            raise NotImplementedError

    @property
    def embed_tokens(self) -> nn.Module:
        if 'gpt' in self.model_id:
            return self.model.transformer.wte
        elif 'opt' in self.model_id:
            return self.model.model.decoder.embed_tokens
        else:
            raise NotImplementedError

    def prepare_inputs_for_generation(self, input_ids, attention_mask, visual_embeds, past_key_values=None, use_cache=None, **kwargs):
        expand_size = input_ids.size(0) // visual_embeds.size(0)
        visual_embeds = visual_embeds.repeat_interleave(expand_size, dim=0)
        visual_mask = torch.ones(visual_embeds.shape[:2], dtype=torch.long, device=visual_embeds.device)

        if input_ids[0][0] == self.model.config.bos_token_id:
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]

        token_embeds = self.embed_tokens(input_ids)
        
        input_embeds = torch.cat([visual_embeds, token_embeds], dim=1)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        input_embeds, attention_mask = accumulate_padding(input_embeds, attention_mask, padding_side='left')

        if past_key_values:
            input_embeds = input_embeds[:, -1].unsqueeze(1)

        return {
            'inputs_embeds': input_embeds,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': use_cache
        }


class MappingNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        output_length: int = 32,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        proj_bias: bool = True
    ) -> None:
        super().__init__()
        self.down = nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        self.up = nn.Linear(hidden_dim, output_dim, bias=proj_bias)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=0.,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.const = nn.Parameter(torch.randn(output_length, hidden_dim))
        # self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = torch.cat((x, self.const.unsqueeze(0).expand(x.size(0), -1, -1)), dim=1)
        x = self.transformer(x)
        x = x[:, -self.const.size(0):]
        x = self.up(x)
        # x = self.norm(x)
        return x


class MAPL(nn.Module):

    def __init__(
        self,
        clip_model_id: str = 'openai/clip-vit-large-patch14',
        gpt_model_id: str = 'EleutherAI/gpt-j-6B'
    ) -> None:
        super().__init__()

        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_id)
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.text_processor = AutoTokenizer.from_pretrained(gpt_model_id)
        if self.text_processor._pad_token is None:
            self.text_processor.pad_token = self.text_processor.eos_token
        self.language_decoder = LanguageDecoder(AutoModelForCausalLM.from_pretrained(gpt_model_id, torch_dtype=torch.float16, revision='float16', low_cpu_mem_usage=True))
        for param in self.language_decoder.parameters():
            param.requires_grad = False

        self.mapper = MappingNetwork(
            input_dim=self.vision_encoder.config.hidden_size,
            output_dim=self.language_decoder.embed_dim
        )
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        dtype: Union[str, torch.dtype] = None,
        **kwargs
    ) -> nn.Module:
        with torch_dtype(dtype):
            model = cls(**kwargs)

        logger.info(f'Loading mapper weights from {checkpoint_path}')
        if is_remote_url(checkpoint_path):
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location='cpu')
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.mapper.load_state_dict(state_dict)

        return model
    
    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            patch_embeds = self.vision_encoder(pixel_values).last_hidden_state
        # patch_embeds = patch_embeds[:, 1:]
        patch_embeds = self.mapper(patch_embeds)
        return patch_embeds
    
    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            token_embeds = self.language_decoder.embed_tokens(input_ids)
        return token_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_ids: torch.Tensor,
        prefix_ids: torch.Tensor = None
    ) -> torch.Tensor:
        visual_embeds = self.embed_image(pixel_values)
        target_embeds = self.embed_text(target_ids)
        if prefix_ids is None:
            input_embeds = torch.cat((visual_embeds, target_embeds), dim=1)
        else:
            prefix_embeds = self.embed_text(prefix_ids)
            input_embeds = torch.cat((visual_embeds, prefix_embeds, target_embeds), dim=1)

        visual_mask = torch.ones(visual_embeds.shape[:2], dtype=torch.long, device=visual_embeds.device)
        target_token_mask = (target_ids != self.text_processor.pad_token_id).long()
        if prefix_ids is None:
            attention_mask = torch.cat((visual_mask, target_token_mask), dim=1)
        else:
            prefix_token_mask = (prefix_ids != self.text_processor.pad_token_id).long()
            attention_mask = torch.cat((visual_mask, prefix_token_mask, target_token_mask), dim=1)
        
        input_embeds, attention_mask = accumulate_padding(input_embeds, attention_mask, padding_side='right')

        outputs = self.language_decoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        return outputs
    
    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor = None,
        **kwargs
    ) -> List[str]:
        visual_embeds = self.embed_image(pixel_values)
        if input_ids is None:
            input_ids = torch.full((visual_embeds.size(0), 1), self.text_processor.bos_token_id, dtype=torch.long, device=visual_embeds.device)
        attention_mask = (input_ids != self.text_processor.pad_token_id).long()

        output_ids = self.language_decoder.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            eos_token_id=self.text_processor.get_vocab()['.'],
            pad_token_id=self.text_processor.pad_token_id,
            **kwargs
        )
        output_ids = output_ids[:, input_ids.size(1):]
        
        return output_ids
    
    def image_transform(self, image: Image.Image, **kwargs) -> torch.Tensor:
        return self.image_processor(image, return_tensors='pt', **kwargs).pixel_values.squeeze(0)
    
    def text_transform(self, text: Union[str, List[str]], **kwargs) -> torch.Tensor:
        return self.text_processor(text, padding='longest', return_tensors='pt', **kwargs)
