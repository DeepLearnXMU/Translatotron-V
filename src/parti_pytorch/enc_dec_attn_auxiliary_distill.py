from typing import Any, Dict, Iterable, List, Optional, Tuple

from functools import partial, wraps

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from parti_pytorch import vit_small, vit_base, vit_tiny, TextTransformerDecoder
from parti_pytorch.t2i_model import T2IBertLayoutTransformer
import timm
from parti_pytorch import VitVQGanVAE, trunc_normal_
from transformers import BertLayer, BertConfig
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

def remove_teacher_model(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_teacer_model = hasattr(self, 'teacher_model')
        if has_teacer_model:
            teacher_model = self.teacher_model
            delattr(self, 'teacher_model')

        out = fn(self, *args, **kwargs)

        if has_teacer_model:
            self.teacher_model = teacher_model

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


class TranslatotronV(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        dropout = 0.1,
        ff_mult = 4,
        image_encoder_weight=None,
        tgt_td_depth=None,
        vae_config=None, 
        vae_weight=None,
        src_text_tokenizer=None,
        tgt_text_tokenizer=None,
        teacher_model_weight=None,
        teacher_model_config=None,
        temperature = 1.,
        cond_drop_prob = 0.,
        vit_mask_prob = 0.,
        ocr_smoothing = 0.1,
        tit_smoothing = 0.1,
        ocr_dropout = 0.1,
        tit_dropout = 0.1,
        **kwargs
    ):
        super().__init__()

        # teacher model initialization
        if teacher_model_config is not None:
            self.teacher_model = T2IBertLayoutTransformer(**teacher_model_config, patch_size = vae_config['patch_size'], img_size = vae_config['image_size'], 
                                                    vae_config = vae_config, vae_weight = vae_weight, tgt_text_tokenizer = tgt_text_tokenizer)
        self.temperature = temperature
        self.cond_drop_prob = cond_drop_prob
        # image conditioning

        self.image_encoder = vit_base(patch_size=vae_config['patch_size'],img_size=vae_config['image_size'], mask_prob = vit_mask_prob)
        
        # vae initialization
        assert exists(vae_config)
        self.vae = VitVQGanVAE(**vae_config)
        
        codebook_size = vae_config['vq_codebook_size']
        image_size = vae_config['image_size']

        
        # tgt text decoder
        self.src_text_tokenizer = src_text_tokenizer
        self.tgt_text_tokenizer = tgt_text_tokenizer
        
        # self.tgt_text_encoder = TextTransformerEncoder()
        if tgt_td_depth is None:
            tgt_td_depth = depth
        self.tgt_text_decoder = TextTransformerDecoder(tgt_text_tokenizer.vocab_size + len(tgt_text_tokenizer.all_special_ids), 
                                                       d_model=dim, ff_size=dim*ff_mult, nhead=heads, num_layers=tgt_td_depth, dropout=tit_dropout,
                                                       kdim=self.image_encoder.embed_dim, vdim=self.image_encoder.embed_dim) 

        self.src_text_decoder = TextTransformerDecoder(src_text_tokenizer.vocab_size + len(src_text_tokenizer.all_special_ids),
                                                       dropout=ocr_dropout,
                                                       kdim=self.image_encoder.embed_dim, vdim=self.image_encoder.embed_dim)
        
        self.start_token = nn.Parameter(torch.randn(dim))
        self.image_token_embed = nn.Embedding(codebook_size, dim)
        
        self.image_encoded_dim = self.vae.get_encoded_fmap_size(image_size)

        self.axial_height_pos = nn.Parameter(torch.randn(self.image_encoded_dim, dim))
        self.axial_width_pos = nn.Parameter(torch.randn(self.image_encoded_dim, dim))

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.src_text_cross_entropy = nn.CrossEntropyLoss(label_smoothing=ocr_smoothing, ignore_index=src_text_tokenizer.pad_token_id)
        self.tgt_text_cross_entropy = nn.CrossEntropyLoss(label_smoothing=tit_smoothing, ignore_index=tgt_text_tokenizer.pad_token_id)
        self.kl_loss = nn.KLDivLoss(reduction='sum')
        # projecting to logits

        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, causal = True, encoded_fmap_size = self.image_encoded_dim, rel_pos_bias = True, dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, context_dim = self.image_encoder.embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, context_dim = self.tgt_text_decoder.d_model, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))
        self.proj_matrix = nn.Linear(2*dim, dim)
        
        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, codebook_size, bias = False)
        # self.to_logits.weight = self.image_token_embed.weight
        
        self.apply(self._init_weights)
        
        if vae_weight is not None:
            self.vae.load_state_dict(torch.load(vae_weight, map_location='cpu'))

        if teacher_model_weight is not None:
            self.teacher_model.load_state_dict(torch.load(teacher_model_weight, map_location='cpu'))
        if image_encoder_weight is not None:
            if "pth" in image_encoder_weight:
                x = torch.load(image_encoder_weight, map_location='cpu')
                del x['pos_embed']
                self.image_encoder.load_state_dict(x, strict=False)
                del x
            elif "npz" in image_encoder_weight:
                x = np.load(image_encoder_weight)
                x = {k: torch.from_numpy(v) for k, v in x.items()}
                del x['pos_embed']
                self.image_encoder.load_state_dict(x, strict=False)
                del x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @remove_vae
    @remove_teacher_model
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vae
    @remove_teacher_model
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    @torch.no_grad()
    @eval_decorator
    def generate_text(
        self,
        src_images,
    ):
        device = next(self.parameters()).device
        
        batch = src_images.shape[0]
        
        max_seq_len = self.image_encoded_dim ** 2
        
        src_text = torch.full((batch, max_seq_len), self.src_text_tokenizer.pad_token_id, dtype=torch.long, device=device)
        
        src_text[:, 0] = self.src_text_tokenizer.bos_token_id
        
        specified_encoder_hidden_state = self.image_encoder.get_specified_layers(src_images, n = [6, 8, 12])
        src_middle_encoder_hidden_states, tgt_middle_encoder_hidden_states, encoder_hidden_states = specified_encoder_hidden_state
        # generate src text
        for cur_len in range(1, max_seq_len):
            tgt_key_padding_mask = src_text[:, :cur_len] == self.src_text_tokenizer.pad_token_id
            causal_mask = self.src_text_decoder.generate_square_subsequent_mask(src_text[:, :cur_len].shape[1]).to(device)
            src_text_logits = self.src_text_decoder(tgt = src_text[:, :cur_len], memory=src_middle_encoder_hidden_states, 
                                                    tgt_mask=causal_mask, tgt_key_padding_mask = tgt_key_padding_mask)
            # greedy decoding
            src_text_sampled = src_text_logits[:,-1].argmax(dim=-1)
            
            src_end_mask = (src_text[:, cur_len - 1] == self.src_text_tokenizer.eos_token_id)
            src_text[:, cur_len] = src_text_sampled
            src_text[src_end_mask, cur_len] = self.src_text_tokenizer.eos_token_id
            
            
            # early stopping
            if (src_text[:, cur_len] == self.src_text_tokenizer.eos_token_id).all():
                break
        
        return src_text
            
    @torch.no_grad()
    def ctc_greedy_decoder(self, log_probs_seq, blank=0):
        """CTC greedy (best path) decoder.

        Path consisting of the most probable tokens are further post-processed to
        remove consecutive repetitions and all blanks.

        Args:
            log_probs_seq: 2-D tensor containing the log probability of a character given each timestep
            blank: blank label index. Defaults to 0
        Returns:
            tuple containing decoded sequence
        """
        # argmax to get the best index for each time step
        max_probs, max_indexes = torch.max(log_probs_seq, -1)
        batch = log_probs_seq.shape[0]
        # remove consecutive duplicate indexes
        mask = torch.cat([
            torch.full((batch, 1), 1, dtype=torch.bool, device=log_probs_seq.device),
            ((max_indexes[:,:-1] - max_indexes[:,1:]).abs() > 0)
        ], dim = 1)
        # remove blank indexes
        mask = mask * (max_indexes != blank)
        max_indexes = max_indexes.masked_fill(~mask, blank)
        return max_indexes

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        # logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """
        # init values
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

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

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

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded

    @torch.no_grad()
    @eval_decorator
    def beam_generate(
        self,
        src_images,
        beam_scorer: BeamScorer,
        num_beams = 4,
        # logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        device = next(self.parameters()).device

        batch = src_images.shape[0]

        image_seq_len = self.image_encoded_dim ** 2

        all_tgt_text_before_output = torch.empty((batch, 0, self.tgt_text_decoder.d_model), 
                                      device = device, dtype = torch.float)
        
        tgt_text = torch.full((batch, image_seq_len), self.tgt_text_tokenizer.pad_token_id, dtype=torch.long, device=device)
        tgt_text[:, 0] = self.src_text_tokenizer.bos_token_id
        
        encoder_hidden_states, middle_encoder_hidden_states = self.image_encoder(src_images, return_middle_layer = 5)
        # generate tgt text first
        for cur_len in range(1,image_seq_len):
            tgt_key_padding_mask = tgt_text[:, :cur_len] == self.tgt_text_tokenizer.pad_token_id
            causal_mask = self.tgt_text_decoder.generate_square_subsequent_mask(tgt_text[:, :cur_len].shape[1]).to(device)
            tgt_text_before_output = self.tgt_text_decoder(tgt = tgt_text[:, :cur_len], memory=encoder_hidden_states, 
                                                    tgt_mask=causal_mask, tgt_key_padding_mask = tgt_key_padding_mask,
                                                    return_before_output_proj=True)
            
            all_tgt_text_before_output = torch.cat((all_tgt_text_before_output, tgt_text_before_output[:,-1].unsqueeze(1)), dim = 1)
            # greedy decoding
            tgt_text_logits = self.tgt_text_decoder.output_layer(tgt_text_before_output)
            tgt_text_sampled = tgt_text_logits[:,-1].argmax(dim=-1)
            
            # judge whether eos or pad
            tgt_end_mask = tgt_text[:, cur_len - 1] == self.tgt_text_tokenizer.eos_token_id
            tgt_pad_mask = tgt_text[:, cur_len - 1] == self.tgt_text_tokenizer.pad_token_id
            tgt_end_mask = tgt_end_mask | tgt_pad_mask
            tgt_text[:, cur_len] = tgt_text_sampled
            tgt_text[tgt_end_mask, cur_len] = self.tgt_text_tokenizer.pad_token_id
            
            # early stopping
            if (tgt_text[:, cur_len] == self.tgt_text_tokenizer.pad_token_id).all():
                break
        
        
        # prepare for image generation
        tgt_text_attn_mask = tgt_text[:,1:cur_len+1] != self.tgt_text_tokenizer.pad_token_id


        input_ids = torch.empty((batch * num_beams, 0), device = device, dtype = torch.long)
        # logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
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

        src_images = src_images.repeat_interleave(num_beams, dim = 0)
        all_tgt_text_before_output = all_tgt_text_before_output.repeat_interleave(num_beams, dim = 0)
        tgt_text_attn_mask = tgt_text_attn_mask.repeat_interleave(num_beams, dim = 0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim = 0)
        middle_encoder_hidden_states = middle_encoder_hidden_states.repeat_interleave(num_beams, dim = 0)
        while cur_len < max_length:

            outputs = self.forward(
                src_images= src_images,
                image_token_ids = input_ids,
                tgt_text_before_output = all_tgt_text_before_output,
                tgt_text_attn_mask = tgt_text_attn_mask,
                encoder_hidden_states = encoder_hidden_states,
                middle_encoder_hidden_states = middle_encoder_hidden_states,
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

        return images, decoded, tgt_text

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        src_images,
        *,
        filter_thres = 0.9,
        temperature = 1.,
        return_pil_images = False
    ):
        device = next(self.parameters()).device

        batch = src_images.shape[0]

        image_seq_len = self.image_encoded_dim ** 2

        image_tokens = torch.empty((batch, 0), device = device, dtype = torch.long)
        all_tgt_text_before_output = torch.empty((batch, 0, self.tgt_text_decoder.d_model), 
                                      device = device, dtype = torch.float)
        
        tgt_text = torch.full((batch, image_seq_len), self.tgt_text_tokenizer.pad_token_id, dtype=torch.long, device=device)
        tgt_text[:, 0] = self.src_text_tokenizer.bos_token_id
        
        encoder_hidden_states, middle_encoder_hidden_states = self.image_encoder(src_images, return_middle_layer = 5)
        # generate tgt text first
        for cur_len in range(1,image_seq_len):
            tgt_key_padding_mask = tgt_text[:, :cur_len] == self.tgt_text_tokenizer.pad_token_id
            causal_mask = self.tgt_text_decoder.generate_square_subsequent_mask(tgt_text[:, :cur_len].shape[1]).to(device)
            tgt_text_before_output = self.tgt_text_decoder(tgt = tgt_text[:, :cur_len], memory=encoder_hidden_states, 
                                                    tgt_mask=causal_mask, tgt_key_padding_mask = tgt_key_padding_mask,
                                                    return_before_output_proj=True)
            
            all_tgt_text_before_output = torch.cat((all_tgt_text_before_output, tgt_text_before_output[:,-1].unsqueeze(1)), dim = 1)
            # greedy decoding
            tgt_text_logits = self.tgt_text_decoder.output_layer(tgt_text_before_output)
            tgt_text_sampled = tgt_text_logits[:,-1].argmax(dim=-1)
            
            # judge whether eos or pad
            tgt_end_mask = tgt_text[:, cur_len - 1] == self.tgt_text_tokenizer.eos_token_id
            tgt_pad_mask = tgt_text[:, cur_len - 1] == self.tgt_text_tokenizer.pad_token_id
            tgt_end_mask = tgt_end_mask | tgt_pad_mask
            tgt_text[:, cur_len] = tgt_text_sampled
            tgt_text[tgt_end_mask, cur_len] = self.tgt_text_tokenizer.pad_token_id
            
            # early stopping
            if (tgt_text[:, cur_len] == self.tgt_text_tokenizer.pad_token_id).all():
                break
        
        
        # prepare for image generation
        tgt_text_attn_mask = tgt_text[:,1:cur_len+1] != self.tgt_text_tokenizer.pad_token_id
        # all_tgt_text_before_output = self.tgt_text_encoder(all_tgt_text_before_output, src_key_padding_mask = ~tgt_text_attn_mask)
        
        for _ in range(image_seq_len):
            logits = self.forward(
                src_images= src_images,
                image_token_ids = image_tokens,
                tgt_text_before_output = all_tgt_text_before_output,
                tgt_text_attn_mask = tgt_text_attn_mask,
                encoder_hidden_states = encoder_hidden_states,
                middle_encoder_hidden_states = middle_encoder_hidden_states,
            )
            logits = logits[:, -1]

            # filtered_logits = top_k(logits, thres = filter_thres)
            # sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

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
            return images, image_tokens, tgt_text

        pil_images = list(map(T.ToPILImage(), images.unbind(dim = 0)))
        return pil_images, image_tokens, tgt_text

    def forward(
        self,
        src_images,
        tgt_images = None,
        image_token_ids = None,
        src_text_input = None,
        tgt_text_input = None,
        tgt_text_before_output = None,
        tgt_text_attn_mask = None,
        encoder_hidden_states = None, 
        middle_encoder_hidden_states = None,
        return_loss = False
    ):
        assert exists(tgt_images) ^ exists(image_token_ids)

        # encoding images

        if not exists(image_token_ids):
            assert exists(self.vae), 'vae must be given if you want to encode the image live'

            with torch.no_grad():
                self.vae.eval()
                _, image_token_ids, _ = self.vae.encode(tgt_images, return_indices_and_loss = True)
                self.vae.training = True

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
        if encoder_hidden_states is None or middle_encoder_hidden_states is None:
            encoder_hidden_states, middle_encoder_hidden_states = self.image_encoder(src_images, return_middle_layer = 5)

        encoder_mask = torch.ones(encoder_hidden_states.shape[:2], dtype = torch.bool, device = device)

        src_text_loss = None
        tgt_text_loss = None
        
        # tgt_text_input can't be None if tgt_text_before_output is None
        assert tgt_text_before_output is not None or tgt_text_input is not None, "tgt_text_input can't be None if tgt_text_before_output is None"
        
        # generate tgt text first
        if tgt_text_before_output is None and tgt_text_attn_mask is None:
            causal_mask = self.tgt_text_decoder.generate_square_subsequent_mask(tgt_text_input["input_ids"].shape[1]).to(device)
            tgt_text_before_output = self.tgt_text_decoder(tgt=tgt_text_input["input_ids"],memory=encoder_hidden_states,
                                                           tgt_mask=causal_mask, tgt_key_padding_mask=~tgt_text_input["attention_mask"].to(device), 
                                                           return_before_output_proj=True)
            tgt_text_output = self.tgt_text_decoder.output_layer(tgt_text_before_output)
            
            # tgt_text_before_output = self.tgt_text_encoder(tgt_text_before_output, src_key_padding_mask = ~tgt_text_input["attention_mask"].to(device))
            
            tgt_text_loss = self.tgt_text_cross_entropy(rearrange(tgt_text_output, 'b n c -> b c n'),
                                                        tgt_text_input["labels"])
            tgt_text_attn_mask = tgt_text_input["attention_mask"]
        # attend

        x = image_token_emb
        
        x = self.init_norm(x)
    

        for i, (self_attn, cross_attn, cross_tgt_attn, ff) in enumerate(self.layers):
            x = self_attn(x) + x
            ix = cross_attn(x, context = encoder_hidden_states, context_mask = encoder_mask)
            tx = cross_tgt_attn(x, context = tgt_text_before_output, context_mask = tgt_text_attn_mask)
            merge_x = torch.cat((tx, ix), dim = -1)
            gate = torch.sigmoid(self.proj_matrix(merge_x))
            x = gate * tx + (1 - gate) * ix + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits

        logits = self.to_logits(x)

        if not return_loss:
            return logits
        
        # teacher model logits
        self.teacher_model.eval()
        teacher_model_logits = self.teacher_model(src_images=src_images, tgt_text_input=tgt_text_input, image_token_ids=image_token_ids)
    
        distill_loss = self.temperature * self.temperature * self.kl_loss(F.log_softmax(logits/self.temperature, dim=-1), F.softmax(teacher_model_logits/self.temperature, dim=-1))/(labels.shape[0]*labels.shape[1])


        loss = self.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels
            )

        # with text auxillary loss
        if src_text_input is not None:
            causal_mask = self.src_text_decoder.generate_square_subsequent_mask(src_text_input["input_ids"].shape[1]).to(device)
            src_text_output = self.src_text_decoder(tgt=src_text_input["input_ids"],memory=middle_encoder_hidden_states,tgt_mask=causal_mask,tgt_key_padding_mask=~src_text_input["attention_mask"].to(device))
            src_text_loss = self.src_text_cross_entropy(rearrange(src_text_output, 'b n c -> b c n'),
                                                        src_text_input["labels"])
    
            
        return loss, distill_loss, src_text_loss, tgt_text_loss
