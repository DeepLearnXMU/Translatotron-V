from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from parti_pytorch import vit_small, vit_base,TextTransformerDecoder, TextTransformerEncoder
import timm
from parti_pytorch import VitVQGanVAE, trunc_normal_
from transformers import BertModel, BertConfig
from .beam_search import BeamScorer

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def remove_vae(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vae = hasattr(self, 'vae')
        if has_vae:
            vae = self.vae
            delattr(self, 'vae')

        out = fn(self, *args, **kwargs)

        if has_vae:
            self.vae = vae

        return out
    return inner

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # account for null key / value for classifier free guidance
        return bias

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        rel_pos_bias = False,
        encoded_fmap_size = None
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

        # positional bias

        self.rel_pos_bias = None

        if rel_pos_bias:
            assert exists(encoded_fmap_size)
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        if exists(self.rel_pos_bias):
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)


class T2IBertLayoutTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        dropout = 0.1,
        ff_mult = 4,
        vae_config=None, 
        vae_weight=None,
        src_text_tokenizer=None,
        tgt_text_tokenizer=None,
        **kwargs
    ):
        super().__init__()

        # vae initialization
        assert exists(vae_config)
        self.vae = VitVQGanVAE(**vae_config)
        self.layout_encoder = timm.create_model("resnet50", pretrained=False)

        codebook_size = vae_config['vq_codebook_size']
        image_size = vae_config['image_size']

        # tgt text decoder
        small_config = {"hidden_size": dim, "hidden_act": "gelu", "initializer_range": 0.02, "vocab_size": tgt_text_tokenizer.vocab_size + len(tgt_text_tokenizer.all_special_ids), 
                        "hidden_dropout_prob": 0.1, "num_attention_heads": heads, "type_vocab_size": 2, "max_position_embeddings": 1024, "num_hidden_layers": depth, 
                        "intermediate_size": dim*ff_mult, "attention_probs_dropout_prob": 0.1}

        self.layout_proj = nn.Linear(2048, small_config["hidden_size"])
        self.bert_config = BertConfig(**small_config)
        self.bert = BertModel(self.bert_config)
        self.tgt_text_tokenizer = tgt_text_tokenizer

        self.start_token = nn.Parameter(torch.randn(dim))
        self.image_token_embed = nn.Embedding(codebook_size, dim)
        
        self.image_encoded_dim = self.vae.get_encoded_fmap_size(image_size)

        self.axial_height_pos = nn.Parameter(torch.randn(self.image_encoded_dim, dim))
        self.axial_width_pos = nn.Parameter(torch.randn(self.image_encoded_dim, dim))

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0)
        # projecting to logits

        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, causal = True, encoded_fmap_size = self.image_encoded_dim, rel_pos_bias = True, dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, context_dim = 2 * self.bert_config.hidden_size, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))
        
        
        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, codebook_size, bias = False)
        
        self.apply(self._init_weights)

        if vae_weight is not None:
            self.vae.load_state_dict(torch.load(vae_weight, map_location='cpu'))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @remove_vae
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vae
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)


    @torch.no_grad()
    @eval_decorator
    def beam_generate(
        self,
        src_images,
        tgt_text_input,
        beam_scorer: BeamScorer,
        num_beams = 4,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        device = next(self.parameters()).device

        batch = tgt_text_input['input_ids'].shape[0]

        image_seq_len = self.image_encoded_dim ** 2
                
        encoder_hidden_states = self.bert(tgt_text_input["input_ids"], attention_mask = tgt_text_input["attention_mask"].to(device))['last_hidden_state']
        layout_hidden_states = self.layout_encoder.forward_features(src_images)
        layout_hidden_states = self.layout_encoder.forward_head(layout_hidden_states, pre_logits = True) 
        layout_hidden_states = self.layout_proj(layout_hidden_states)


        input_ids = torch.empty((batch * num_beams, 0), device = device, dtype = torch.long)
        max_length = max_length if max_length is not None else self.image_encoded_dim ** 2
        pad_token_id = -1
        eos_token_id = -2

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim = 0)
        layout_hidden_states = layout_hidden_states.repeat_interleave(num_beams, dim = 0)
        tgt_text_input['input_ids'] = tgt_text_input['input_ids'].repeat_interleave(num_beams, dim = 0)
        tgt_text_input['attention_mask'] = tgt_text_input['attention_mask'].repeat_interleave(num_beams, dim = 0)

        while cur_len < max_length:

            outputs = self.forward(
                tgt_text_input= tgt_text_input,
                image_token_ids = input_ids,
                layout_hidden_states=layout_hidden_states,
                encoder_hidden_states = encoder_hidden_states
            )
            next_token_logits = outputs[:, -1, :]

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        decoded = rearrange(decoded, 'b (h w) -> b h w', h = self.image_encoded_dim)
        
        if not exists(self.vae):
            return decoded

        with torch.no_grad():
            fmap = self.vae.get_fmap_from_codebook(decoded)
            images = self.vae.decode(fmap)

        return images, decoded
    
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        src_images,
        tgt_text_input,
        *,
        filter_thres = 0.9,
        temperature = 1.,
        return_pil_images = False
    ):
        device = next(self.parameters()).device

        batch = tgt_text_input['input_ids'].shape[0]

        image_seq_len = self.image_encoded_dim ** 2

        image_tokens = torch.empty((batch, 0), device = device, dtype = torch.long)
                
        encoder_hidden_states = self.bert(tgt_text_input["input_ids"], attention_mask = tgt_text_input["attention_mask"].to(device))['last_hidden_state']
        layout_hidden_states = self.layout_encoder.forward_features(src_images)
        layout_hidden_states = self.layout_encoder.forward_head(layout_hidden_states, pre_logits = True) 
        layout_hidden_states = self.layout_proj(layout_hidden_states) # shape (batch, 1, hidden_size)

        for _ in range(image_seq_len):
            logits = self.forward(
                tgt_text_input= tgt_text_input,
                image_token_ids = image_tokens,
                layout_hidden_states=layout_hidden_states,
                encoder_hidden_states = encoder_hidden_states
            )
            logits = logits[:, -1]

            #greedy search
            sampled = logits.argmax(dim=-1)
            sampled = rearrange(sampled, 'b -> b 1')
            image_tokens = torch.cat((image_tokens, sampled), dim = -1)             
        
        image_tokens = rearrange(image_tokens, 'b (h w) -> b h w', h = self.image_encoded_dim)
        
        if not exists(self.vae):
            return image_tokens

        with torch.no_grad():
            fmap = self.vae.get_fmap_from_codebook(image_tokens)
            images = self.vae.decode(fmap)

        if not return_pil_images:
            return images, image_tokens

        pil_images = list(map(T.ToPILImage(), images.unbind(dim = 0)))
        return pil_images, image_tokens

    def forward(
        self,
        src_images = None,
        tgt_text_input = None,
        tgt_images = None,
        image_token_ids = None,
        layout_hidden_states = None,
        encoder_hidden_states = None,
        return_loss = False
    ):
        assert exists(tgt_images) ^ exists(image_token_ids)

        # encoding images

        if not exists(image_token_ids):
            assert exists(self.vae), 'vae must be given if you want to encode the image live'

            with torch.no_grad():
                self.vae.eval()
                _, image_token_ids, _ = self.vae.encode(tgt_images, return_indices_and_loss = True)

            image_token_ids = rearrange(image_token_ids, 'b ... -> b (...)')

        if return_loss:
            assert image_token_ids.shape[-1] > 1, 'not enough image tokens given to return a loss'
            image_token_ids, labels = image_token_ids[:, :-1], image_token_ids

        image_token_emb = self.image_token_embed(image_token_ids)

        # add axial positional embedding

        axial_pos_emb = rearrange(self.axial_width_pos, 'w d -> 1 w d') + rearrange(self.axial_height_pos, 'h d -> h 1 d')
        axial_pos_emb = rearrange(axial_pos_emb, 'h w d -> (h w) d')

        batch, seq_len, device = *image_token_emb.shape[:2], image_token_emb.device

        image_token_emb = image_token_emb + axial_pos_emb[:seq_len]

        # add start token

        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = batch)
        image_token_emb = torch.cat((start_tokens, image_token_emb), dim = 1)

        # encoder hidden states
        if encoder_hidden_states is None:
            encoder_hidden_states = self.bert(tgt_text_input["input_ids"], attention_mask = tgt_text_input["attention_mask"].to(device))['last_hidden_state']
        

        encoder_mask = tgt_text_input["attention_mask"]
        
        # add layout hidden states
        if layout_hidden_states is None:
            layout_hidden_states = self.layout_encoder.forward_features(src_images)
            layout_hidden_states = self.layout_encoder.forward_head(layout_hidden_states, pre_logits = True) 
            layout_hidden_states = self.layout_proj(layout_hidden_states) # shape (batch, hidden_size)
        
        layout_hidden_states = repeat(layout_hidden_states, 'b d -> b n d', n = encoder_hidden_states.shape[1])
        # concat layout hidden states to each encoder hidden states
        encoder_hidden_states = torch.cat((layout_hidden_states, encoder_hidden_states), dim = -1) # shape (batch, seq_len, 2 * hidden_size)
        x = image_token_emb

        x = self.init_norm(x)
        
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x) + x
            x = cross_attn(x, context = encoder_hidden_states, context_mask = encoder_mask) + x
            x = ff(x) + x


        x = self.final_norm(x)

        # to logits

        logits = self.to_logits(x)

        if not return_loss:
            return logits
        
        loss = self.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels
            )
            
        return loss