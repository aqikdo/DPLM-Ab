# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
from byprot.models.dplm2.modules.dplm2_modeling_esm import *
from byprot.models.utils import *


def exists(obj):
    return obj is not None


@dataclass
class SelfMixupConfig:
    enable: bool = field(default=False)
    with_original_loss: bool = field(default=False)


@dataclass
class TokenizerConfig:
    vocab_file: str = field(default="airkingbd/dplm2_650m")
    # amino acid tokens (33) + struct tokens (8192) + 4 special struct tokens
    vocab_size: int = field(default=33 + 8192 + 4)


@dataclass
class StructTokenizerConfig:
    enable: bool = field(default=True)
    exp_path: str = field(default="airkingbd/struct_tokenizer")


@dataclass
class MixedTrainingConfig:
    enable: bool = field(default=False)


@dataclass
class AntigenConditionConfig:
    enable: bool = field(default=False)
    freeze_encoder: bool = field(default=True)
    epitope_loss_weight: float = field(default=1.0)
    # After Ag encoder, add a learned 0/1 epitope embedding onto Ag hidden
    # states that feed Ab→Ag cross-attn. Full Ag (pad mask only) remains visible;
    # epitope is an extra binary feature, not a key-dropping mask.
    epitope_cross_attn_feature: bool = field(default=False)
    # Deprecated / unused: previous hard key-mask experiment. Kept so old
    # configs still load; do not re-enable.
    epitope_cross_attn_mask: bool = field(default=False)
    # If True, cross-attn out_proj is zero-init (identity at start).
    # If False, use default Linear init so antigen signal is present immediately.
    cross_attn_zero_init: bool = field(default=True)
    # Multiply base LR for conditional cross-attn params (aggressive Ag injection).
    cross_attn_lr_scale: float = field(default=1.0)
    # Epitope predictor: TransformerDecoder over Ag tokens, cross-attending to Ab
    # last_hidden_state (pre-lm_head, struct+aa).
    epitope_transformer_layers: int = field(default=2)
    epitope_transformer_heads: int = field(default=8)
    epitope_transformer_dropout: float = field(default=0.1)
    net: NetConfig = field(default=NetConfig())


class EpitopeTransformerPredictor(nn.Module):
    """Predict Ag-residue epitope from Ag encoder states + Ab pre-lm_head states.

    Ag aa-token hidden states are queries; Ab backbone last_hidden_state
    (struct+aa, after conditional cross-attn) is memory / K,V.
    """

    def __init__(
        self,
        hidden_size: int,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ag_norm = nn.LayerNorm(hidden_size)
        self.ab_norm = nn.LayerNorm(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        ag_hidden: torch.Tensor,
        ab_hidden: torch.Tensor,
        ag_key_padding_mask: Optional[torch.Tensor] = None,
        ab_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt = self.ag_norm(ag_hidden)
        memory = self.ab_norm(ab_hidden)
        hidden = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=ag_key_padding_mask,
            memory_key_padding_mask=ab_key_padding_mask,
        )
        return self.out_proj(self.out_norm(hidden)).squeeze(-1)


@dataclass
class CdrGenerationConfig:
    """Only noise/supervise CDR residues; keep framework aa+struct clean."""

    enable: bool = field(default=False)


@dataclass
class MaskScheduleConfig:
    """Curriculum on mask probability: scale * (t / T).

    Spatial CDR restriction is controlled by ``cdr_generation`` and/or
    ``cdr_noise_ratio`` (fraction of each batch that uses CDR-only + progressive
    scale; the rest uses original full-sequence masking at scale=1.0).
    """

    enable: bool = field(default=False)
    warmup_steps: int = field(default=1500)
    min_scale: float = field(default=0.1)
    max_scale: float = field(default=1.0)
    schedule: str = field(default="linear")  # linear | cosine
    # 0.0 = legacy (all samples follow cdr_generation flag);
    # (0, 1] = mix: this fraction uses CDR-only + progressive scale,
    #         remainder uses original full masking @ scale=1.0.
    cdr_noise_ratio: float = field(default=0.0)


@dataclass
class TwoStageTrainingConfig:
    enable: bool = field(default=False)
    stage1_steps: int = field(default=1000)
    stage2_lr_scale: float = field(default=0.1)
    freeze_backbone: bool = field(default=True)
    freeze_antigen_encoder: bool = field(default=True)
    # When antigen encoder is trainable, scale its LR relative to base LR.
    antigen_encoder_lr_scale: float = field(default=1.0)


@dataclass
class DPLM2Config:
    ## DPLM model
    num_diffusion_timesteps: int = field(default=500)
    tokenizer: TokenizerConfig = field(default=TokenizerConfig())
    lora: LoRAConfig = field(default=LoRAConfig())
    net: NetConfig = field(default=NetConfig())
    gradient_ckpt: bool = field(default=False)

    ## multi-modal training
    training_stage: str = field(default="train_from_dplm")
    self_mixup: SelfMixupConfig = field(
        default=SelfMixupConfig()
    )  # training strategy
    single_modality_ratio: float = field(default=0.25)
    folding_loss_ratio: float = field(default=0.25)
    inverse_folding_loss_ratio: float = field(default=0.25)
    joint_loss_ratio: float = field(default=0.25)
    independent_loss_ratio: float = field(default=0.0)
    zero_struct_loss: bool = field(default=False)
    mixed_training: MixedTrainingConfig = field(default=MixedTrainingConfig())

    ## finetune task: null (full multimodal) | sequence_generation | backbone_generation
    finetune_task: Optional[str] = field(default=None)

    ## optional finetune freezes (toggle via yaml / CLI overrides)
    freeze_word_embeddings: bool = field(default=False)
    freeze_position_embeddings: bool = field(default=False)
    freeze_lm_head_decoder: bool = field(default=False)
    freeze_lm_head: bool = field(default=False)
    freeze_encoder: bool = field(default=False)

    ## struct tokenizer
    struct_tokenizer: StructTokenizerConfig = field(
        default=StructTokenizerConfig()
    )
    antigen_condition: AntigenConditionConfig = field(
        default=AntigenConditionConfig()
    )
    two_stage: TwoStageTrainingConfig = field(default=TwoStageTrainingConfig())
    cdr_generation: CdrGenerationConfig = field(default=CdrGenerationConfig())
    mask_schedule: MaskScheduleConfig = field(default=MaskScheduleConfig())


@register_model("dplm2")
class MultimodalDiffusionProteinLanguageModel(nn.Module):
    _default_cfg = DPLM2Config()

    def __init__(self, cfg, net=None):
        super().__init__()
        self._update_cfg(cfg)
        self.tokenizer = DPLM2Tokenizer.from_pretrained(
            self.cfg.tokenizer.vocab_file
        )
        self._prepare_special_token()
        self.cfg.tokenizer.vocab_size = len(self.tokenizer)
        if net is None:
            self.net = get_net_dplm2(self.cfg)
        else:
            if "bit" in net.config.dplm_type:
                raise ValueError(
                    f"Bit model is not supported in this DPLM-2 class, please use DPLM-2 bit model instead."
                )
            self.net = net

        if self.cfg.gradient_ckpt:
            self.net.supports_gradient_checkpointing = True
            self.net.gradient_checkpointing_enable()
            # Peft / frozen-embedding + gradient checkpointing drops all grads
            # unless at least one input tensor requires grad.
            self._ensure_input_require_grads()

        self.antigen_encoder = self._build_antigen_encoder()
        if (
            self.cfg.gradient_ckpt
            and exists(self.antigen_encoder)
            and hasattr(self.antigen_encoder, "gradient_checkpointing_enable")
        ):
            self.antigen_encoder.supports_gradient_checkpointing = True
            self.antigen_encoder.gradient_checkpointing_enable()
        self.epitope_head = self._build_epitope_head()
        self.epitope_feature_embed = self._build_epitope_feature_embed()
        self._struct_tokenizer = None
        self._conditional_stage = None
        self.apply_parameter_freeze()

    def _ensure_input_require_grads(self):
        """Make encoder inputs require grad so checkpointing keeps adapter grads.

        With Peft, `modules_to_save` often wraps the whole `esm.embeddings` module.
        DPLM2 also builds `inputs_embeds` via `self.net.esm.embeddings(...)` before
        the encoder, so the hook must sit on that embeddings module.
        """
        if getattr(self, "_input_require_grads_hooked", False):
            return

        embeddings_mod = None
        net = self.net
        try:
            esm = getattr(net, "esm", None)
            if esm is None:
                base = getattr(net, "base_model", None)
                model = getattr(base, "model", base) if base is not None else None
                esm = getattr(model, "esm", None) if model is not None else None
            if esm is not None and hasattr(esm, "embeddings"):
                embeddings_mod = esm.embeddings
        except Exception:
            embeddings_mod = None
        if embeddings_mod is None:
            return

        def _hook(module, inp, out):
            if torch.is_tensor(out):
                return out.requires_grad_(True)
            return out

        embeddings_mod.register_forward_hook(_hook)
        self._input_require_grads_hooked = True

    def apply_parameter_freeze(self):
        if self.cfg.freeze_encoder:
            for p in self.net.esm.encoder.parameters():
                p.requires_grad = False
        if self.cfg.freeze_word_embeddings:
            self.net.esm.embeddings.word_embeddings.weight.requires_grad = False
        if self.cfg.freeze_position_embeddings:
            self.net.esm.embeddings.position_embeddings.weight.requires_grad = False
        if self.cfg.freeze_lm_head_decoder:
            self.net.lm_head.decoder.weight.requires_grad = False
        if self.cfg.freeze_lm_head:
            for p in self.net.lm_head.parameters():
                p.requires_grad = False
        if (
            exists(self.antigen_encoder)
            and getattr(self.cfg.antigen_condition, "freeze_encoder", False)
        ):
            for p in self.antigen_encoder.parameters():
                p.requires_grad = False

    def _update_cfg(self, cfg):
        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    @property
    def special_token_list(self):
        return [
            self.aa_bos_id,
            self.aa_eos_id,
            self.aa_mask_id,
            self.struct_bos_id,
            self.struct_eos_id,
            self.struct_mask_id,
            self.pad_id,
            self.aa_unk_id,
            self.struct_unk_id,
            self.aa_X_id,
            self.aa_B_id,
            self.aa_U_id,
            self.aa_Z_id,
            self.aa_O_id,
        ]

    @classmethod
    def from_pretrained(
        cls, net_name, cfg_override={}, net_override={}, from_huggingface=True
    ):
        if str(net_name).endswith(".ckpt"):
            from_huggingface = False
        if not from_huggingface:
            # Load model checkpoint from local if you pretrain a DPLM with this repo
            # The net_name should be like:
            # ${name}/checkpoints/last.ckpt
            # and there should be .hydra/config.yaml in the ${name} directory that is automatically generated during training.
            from collections import OrderedDict
            from pathlib import Path

            from byprot.utils.config import load_yaml_config

            cfg_path = None
            for parent in Path(net_name).parents:
                candidate = parent / ".hydra" / "config.yaml"
                if candidate.is_file():
                    cfg_path = candidate
                    break
            if cfg_path is None:
                raise FileNotFoundError(
                    f"No .hydra/config.yaml found for checkpoint: {net_name}"
                )
            cfg = load_yaml_config(str(cfg_path))
            OmegaConf.resolve(cfg)
            cfg = cfg.model
            cfg.net.pretrain = False
            cfg.pop("_target_")

            model = cls(cfg)

            pretrained_state_dict = torch.load(
                net_name, map_location=torch.device("cpu")
            )["state_dict"]
            new_pretrained_state_dict = OrderedDict()

            # remove the module prefix "model."
            for k, v in pretrained_state_dict.items():
                new_pretrained_state_dict[k[6:]] = v
            missing, unexpected = model.load_state_dict(
                new_pretrained_state_dict, strict=False
            )
            print(
                f"Restored from {net_name} with {len(missing)} missing and {len(unexpected)} unexpected keys"
            )
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
                print(f"Unexpected Keys: {unexpected}")
            return model

        else:
            # Load DPLM-2 model checkpoint from huggingface
            dplm_type = AutoConfig.from_pretrained(net_name).dplm_type
            net_class = get_net_class(dplm_type)
            net = net_class.from_pretrained(net_name, **net_override)
            return cls(cfg=cfg_override, net=net)

    def _prepare_special_token(self):
        self.aa_bos_id = self.tokenizer._token_to_id["<cls_aa>"]
        self.aa_eos_id = self.tokenizer._token_to_id["<eos_aa>"]
        self.aa_mask_id = self.tokenizer._token_to_id["<mask_aa>"]
        self.struct_bos_id = self.tokenizer._token_to_id["<cls_struct>"]
        self.struct_eos_id = self.tokenizer._token_to_id["<eos_struct>"]
        self.struct_mask_id = self.tokenizer._token_to_id["<mask_struct>"]
        self.pad_id = self.tokenizer._token_to_id["<pad>"]
        self.aa_unk_id = self.tokenizer._token_to_id["<unk_aa>"]
        self.struct_unk_id = self.tokenizer._token_to_id["<unk_struct>"]

        self.aa_X_id = self.tokenizer._token_to_id["X"]
        self.aa_B_id = self.tokenizer._token_to_id["B"]
        self.aa_U_id = self.tokenizer._token_to_id["U"]
        self.aa_Z_id = self.tokenizer._token_to_id["Z"]
        self.aa_O_id = self.tokenizer._token_to_id["O"]

        self.aa_type = 1
        self.struct_type = 0
        self.pad_type = 2

    def _build_antigen_encoder(self):
        if not getattr(self.cfg.antigen_condition, "enable", False):
            return None
        antigen_net_cfg = OmegaConf.to_container(
            self.cfg.antigen_condition.net, resolve=True
        )
        if not antigen_net_cfg.get("name"):
            antigen_net_cfg["name"] = self.cfg.net.name
        if not antigen_net_cfg.get("pretrained_model_name_or_path"):
            antigen_net_cfg["pretrained_model_name_or_path"] = (
                self.cfg.net.pretrained_model_name_or_path or self.cfg.net.name
            )
        if not antigen_net_cfg.get("pretrain"):
            antigen_net_cfg["pretrain"] = True
        antigen_cfg = OmegaConf.create(
            {
                "training_stage": self.cfg.training_stage,
                "net": antigen_net_cfg,
                "tokenizer": {"vocab_size": len(self.tokenizer)},
                "lora": {"enable": False},
                "antigen_condition": {"enable": False},
            }
        )
        return get_net_dplm2(antigen_cfg)

    def _build_epitope_head(self):
        if not getattr(self.cfg.antigen_condition, "enable", False):
            return None
        cfg = self.cfg.antigen_condition
        # epitope_loss_weight<=0 means disable epitope head entirely
        if float(getattr(cfg, "epitope_loss_weight", 1.0)) <= 0.0:
            return None
        hidden = self.net.config.hidden_size
        return EpitopeTransformerPredictor(
            hidden_size=hidden,
            nhead=int(getattr(cfg, "epitope_transformer_heads", 8)),
            num_layers=int(getattr(cfg, "epitope_transformer_layers", 2)),
            dropout=float(getattr(cfg, "epitope_transformer_dropout", 0.1)),
        )

    def _build_epitope_feature_embed(self):
        """Learned 0/1 embedding added to Ag encoder states for cross-attn."""
        if not getattr(self.cfg.antigen_condition, "enable", False):
            return None
        if not bool(
            getattr(
                self.cfg.antigen_condition, "epitope_cross_attn_feature", False
            )
        ):
            return None
        hidden = self.net.config.hidden_size
        embed = nn.Embedding(2, hidden)
        # Zero-init → starts as no-op; model learns epitope bias from data.
        nn.init.zeros_(embed.weight)
        return embed

    def predict_epitopes(
        self,
        antigen_hidden_states,
        antibody_hidden_states=None,
        antigen_attention_mask=None,
        antibody_attention_mask=None,
    ):
        if not exists(self.epitope_head) or antigen_hidden_states is None:
            return None
        if antibody_hidden_states is None:
            return None
        # Antigen encoder concat is [struct | aa]; epitope labels align to aa half.
        if antigen_hidden_states.size(1) % 2 == 0:
            _, antigen_hidden_states = antigen_hidden_states.chunk(2, dim=1)
            if antigen_attention_mask is not None:
                _, antigen_attention_mask = antigen_attention_mask.chunk(2, dim=1)
        ag_pad = (
            ~antigen_attention_mask.bool()
            if antigen_attention_mask is not None
            else None
        )
        ab_pad = (
            ~antibody_attention_mask.bool()
            if antibody_attention_mask is not None
            else None
        )
        return self.epitope_head(
            antigen_hidden_states,
            antibody_hidden_states,
            ag_key_padding_mask=ag_pad,
            ab_key_padding_mask=ab_pad,
        )

    @property
    def device(self):
        try:
            device = next(self.parameters()).device
        except:
            device = torch.device("cpu")
        return device

    @property
    def struct_tokenizer(self):
        if not exists(self._struct_tokenizer):
            print(f"Loading struct_tokenizer...")
            self._struct_tokenizer = get_struct_tokenizer(
                self.cfg.struct_tokenizer.exp_path
            ).to(self.device)
        return self._struct_tokenizer

    def _mask_scale(self, global_step: int = 0) -> float:
        """Return curriculum scale for mask probability (relative to t/T)."""
        cfg = getattr(self.cfg, "mask_schedule", None)
        if cfg is None or not bool(getattr(cfg, "enable", False)):
            return 1.0
        warmup = max(int(getattr(cfg, "warmup_steps", 1500)), 1)
        min_s = float(getattr(cfg, "min_scale", 0.1))
        max_s = float(getattr(cfg, "max_scale", 1.0))
        progress = min(1.0, float(global_step) / float(warmup))
        schedule = str(getattr(cfg, "schedule", "linear")).lower()
        if schedule == "cosine":
            # Smooth ramp from min to max over warmup.
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return min_s + (max_s - min_s) * progress

    def q_sample(self, x_0, t, type_ids, maskable_mask, mask_scale: float = 1.0):
        aa_position = type_ids == self.aa_type
        struct_position = type_ids == self.struct_type

        # sample x_t; mask_scale curricula the Bernoulli probability vs t/T
        u = torch.rand_like(x_0, dtype=torch.float)
        scale = float(mask_scale)
        t_mask = (
            u < (t.float() / self.cfg.num_diffusion_timesteps * scale)[:, None]
        ) & maskable_mask
        x_t = x_0.masked_fill(t_mask & aa_position, self.aa_mask_id)
        x_t = x_t.masked_fill(t_mask & struct_position, self.struct_mask_id)

        return x_t, t_mask

    def get_modality_type(self, input_ids):
        input_mask = input_ids.ne(self.pad_id)
        # HACK: all amino acid token id < 33, while all struct token id >= 33
        # 0 stands for struct, 1 stands for aa
        modality_type = ((input_ids < 33) & input_mask).int()
        # 2 stands for padding
        modality_type[~input_mask] = self.pad_type
        return modality_type

    def _build_attention_bias(self, input_ids, single_modality=None):
        input_mask = input_ids.ne(self.pad_id)
        L = input_ids.shape[1]
        num_heads = self.net.config.num_attention_heads
        attention_bias: torch.FloatType = (
            self.net.esm.get_extended_attention_mask(
                input_mask, input_ids.shape
            ).repeat(1, num_heads, L, 1)
        )
        if single_modality is not None:
            struct_attention_bias, aa_attention_bias = attention_bias.chunk(
                2, dim=-2
            )
            struct_attention_bias[
                single_modality, :, :, L // 2 :
            ] = -math.inf
            aa_attention_bias[
                single_modality, :, :, : L // 2
            ] = -math.inf
            attention_bias = torch.concat(
                [struct_attention_bias, aa_attention_bias], dim=-2
            )
        return input_mask, attention_bias

    def _concat_modal_tokens(self, struct_tokens, aatype_tokens):
        return torch.concat([struct_tokens, aatype_tokens], dim=1)

    def _epitope_ids_for_cross_attn(self, antigen_batch, aa_len, device):
        """Build [B, 2*aa_len] long ids (0/1) aligned to Ag [struct|aa].

        epitope_labels are aa-half length (incl. CLS/EOS). Missing labels → all 0.
        """
        labels = antigen_batch.get("epitope_labels")
        if labels is None:
            return torch.zeros(
                antigen_batch["aatype_tokens"]["targets"].size(0),
                aa_len * 2,
                dtype=torch.long,
                device=device,
            )
        epi = labels.to(device=device, dtype=torch.float)
        if epi.dim() == 1:
            epi = epi.unsqueeze(0)
        bsz = epi.size(0)
        if epi.size(1) < aa_len:
            epi = torch.nn.functional.pad(epi, (0, aa_len - epi.size(1)))
        elif epi.size(1) > aa_len:
            epi = epi[:, :aa_len]
        epi_ids = (epi > 0.5).long()
        return torch.cat([epi_ids, epi_ids], dim=1)

    def _apply_epitope_feature(self, hidden, antigen_batch):
        """Add 0/1 epitope embedding onto Ag hidden states (full Ag still visible)."""
        if not exists(self.epitope_feature_embed):
            return hidden
        bsz, full_len, _ = hidden.shape
        if full_len % 2 != 0:
            return hidden
        aa_len = full_len // 2
        epi_ids = self._epitope_ids_for_cross_attn(
            antigen_batch, aa_len, hidden.device
        )
        if epi_ids.size(0) != bsz:
            return hidden
        return hidden + self.epitope_feature_embed(epi_ids)

    def encode_antigen(self, antigen_batch):
        if not exists(self.antigen_encoder) or antigen_batch is None:
            return None
        antigen_input_ids = self._concat_modal_tokens(
            antigen_batch["struct_tokens"]["targets"],
            antigen_batch["aatype_tokens"]["targets"],
        )
        antigen_mask, antigen_attention_bias = self._build_attention_bias(
            antigen_input_ids
        )
        antigen_type_ids = self.get_modality_type(antigen_input_ids)
        antigen_inputs_embeds = self.antigen_encoder.esm.embeddings(
            antigen_input_ids, attention_mask=antigen_mask
        )
        outputs = self.antigen_encoder.esm(
            input_ids=antigen_input_ids,
            inputs_embeds=antigen_inputs_embeds,
            attention_mask=antigen_attention_bias,
            type_ids=antigen_type_ids,
        )
        hidden = outputs.last_hidden_state
        has_antigen = antigen_batch.get("has_antigen")
        if has_antigen is not None:
            hidden = hidden * has_antigen.to(hidden.device)[:, None, None].float()
        # Extra 0/1 epitope feature only for Ab→Ag cross-attn keys/values.
        # epitope_head still scores raw encoder states over full Ag.
        hidden_for_xattn = self._apply_epitope_feature(hidden, antigen_batch)
        return {
            "input_ids": antigen_input_ids,
            "attention_mask": antigen_mask,
            "pad_attention_mask": antigen_mask,
            "hidden_states": hidden_for_xattn,
            "encoder_hidden_states": hidden,
        }

    def forward(self, input_ids, **kwargs):
        type_ids = self.get_modality_type(input_ids)
        single_modality_index = kwargs.get("single_modality")
        input_mask, attention_bias = self._build_attention_bias(
            input_ids, single_modality=single_modality_index
        )

        # [B, L, d_model]
        input_embeds = self.net.esm.embeddings(
            input_ids, attention_mask=input_mask
        )

        outputs = self.net(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_bias,
            type_ids=type_ids,
            conditional_hidden_states=kwargs.get("antigen_hidden_states"),
            conditional_attention_mask=kwargs.get("antigen_attention_mask"),
        )

        return outputs

    def self_mixup(self, x_t, single_modality_index):
        # 1. first part: masked prediction
        with torch.no_grad():
            model_outputs = self.forward(
                input_ids=x_t, single_modality=single_modality_index
            )
            lm_logits = model_outputs["logits"]
        # 2. mixup: alternate mask with model prediction and gt with masks
        prev_input_ids = x_t
        non_special_sym_mask = self.get_non_special_symbol_mask(prev_input_ids)
        model_pred = torch.where(
            non_special_sym_mask, lm_logits.argmax(dim=-1), prev_input_ids
        )
        mixup_xt, mixup_loss_mask = self.get_mixup_xt(
            input_ids=prev_input_ids,
            model_pred=model_pred,
            non_special_sym_mask=non_special_sym_mask,
        )

        # # 3. second part: denoising + masked prediction
        model_outputs = self.forward(
            input_ids=mixup_xt, single_modality=single_modality_index
        )
        return model_outputs, mixup_loss_mask

    def get_mixup_xt(self, input_ids, model_pred, non_special_sym_mask=None):
        gt_mask = (
            input_ids.ne(self.aa_mask_id)
            & input_ids.ne(self.struct_mask_id)
            & non_special_sym_mask
        )

        type_ids = self.get_modality_type(input_ids)

        mixup_input_ids = model_pred
        # replace gt positions with mask
        mixup_input_ids = mixup_input_ids.masked_fill(
            gt_mask & (type_ids == self.aa_type), self.aa_mask_id
        )
        mixup_input_ids = mixup_input_ids.masked_fill(
            gt_mask & (type_ids == self.struct_type), self.struct_mask_id
        )
        mixup_loss_mask = non_special_sym_mask
        return mixup_input_ids, mixup_loss_mask

    def _mixed_training_enabled(self) -> bool:
        return bool(getattr(self.cfg, "mixed_training", None) and self.cfg.mixed_training.enable)

    def _zero_struct_loss_enabled(self) -> bool:
        return bool(getattr(self.cfg, "zero_struct_loss", False))

    def _construct_x_t_with_targets(
        self, struct_target, aatype_target, cdr_mask=None, mask_scale: float = 1.0
    ):
        bsz = struct_target.size(0)
        struct_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=struct_target.device,
        )
        aatype_t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=aatype_target.device,
        )

        split_sizes = [
            int(bsz * self.cfg.single_modality_ratio),
            int(bsz * self.cfg.folding_loss_ratio),
            int(bsz * self.cfg.inverse_folding_loss_ratio),
            int(bsz * self.cfg.independent_loss_ratio),
            int(bsz * self.cfg.joint_loss_ratio),
        ]
        split_sizes[-1] = bsz - sum(split_sizes[:-1])

        rand_index = torch.randperm(bsz).type_as(struct_target)
        int_index_list = torch.split(rand_index, split_sizes)

        bool_index_list = []
        for int_index in int_index_list:
            bool_index = torch.zeros(bsz, dtype=torch.bool, device=struct_target.device)
            bool_index[int_index] = True
            bool_index_list.append(bool_index)

        (
            single_modality_index,
            folding_index,
            inverse_folding_index,
            independent_index,
            joint_index,
        ) = bool_index_list

        struct_t = struct_t.masked_fill(inverse_folding_index, 0)
        struct_type_id = self.get_modality_type(struct_target)
        struct_x_t, struct_loss_mask = self.q_sample(
            struct_target,
            struct_t,
            struct_type_id,
            maskable_mask=self._cdr_maskable_mask(struct_target, cdr_mask),
            mask_scale=mask_scale,
        )
        aatype_t = aatype_t.masked_fill(folding_index, 0)
        aatype_t = aatype_t.masked_scatter(joint_index, struct_t[joint_index])
        aa_type_id = self.get_modality_type(aatype_target)
        aatype_x_t, aa_loss_mask = self.q_sample(
            aatype_target,
            aatype_t,
            aa_type_id,
            maskable_mask=self._cdr_maskable_mask(aatype_target, cdr_mask),
            mask_scale=mask_scale,
        )

        return (
            {"t": struct_t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": aatype_t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    def _construct_x_t_seq_only_dummy(self, struct_target, aatype_target):
        """Match dummy_struct + zero_struct_loss + single_modality_ratio=1.0.

        Both halves are noised; struct loss is dropped; AA cannot attend to
        struct (single_modality). Used for every NGS row in mixed training so
        sr=0 is equivalent to the legacy pure-seq recipe.
        """
        bsz = struct_target.size(0)
        device = struct_target.device
        num_timesteps = self.cfg.num_diffusion_timesteps
        struct_t = torch.randint(
            1, num_timesteps + 1, (bsz,), device=device
        )
        aatype_t = torch.randint(
            1, num_timesteps + 1, (bsz,), device=device
        )
        single_modality_index = torch.ones(bsz, dtype=torch.bool, device=device)

        struct_x_t, struct_loss_mask = self.q_sample(
            struct_target,
            struct_t,
            self.get_modality_type(struct_target),
            maskable_mask=self.get_non_special_symbol_mask(struct_target),
        )
        aatype_x_t, aa_loss_mask = self.q_sample(
            aatype_target,
            aatype_t,
            self.get_modality_type(aatype_target),
            maskable_mask=self.get_non_special_symbol_mask(aatype_target),
        )
        # Never supervise dummy/placeholder struct on NGS.
        struct_loss_mask = torch.zeros_like(struct_loss_mask)
        return (
            {"t": struct_t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": aatype_t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    def _construct_x_t_mixed(self, struct_target, aatype_target, is_ngs, mask_scale: float = 1.0):
        """Mixed NGS+struct noising.

        Struct rows: full 4-way multimodal objective.
        NGS rows: identical to dummy pure-seq
        (``_construct_x_t_seq_only_dummy``).
        """
        bsz = struct_target.size(0)
        device = struct_target.device
        struct_t = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1, (bsz,), device=device
        )
        aatype_t = torch.randint(
            1, self.cfg.num_diffusion_timesteps + 1, (bsz,), device=device
        )
        single_modality_index = torch.ones(bsz, dtype=torch.bool, device=device)

        struct_x_t = struct_target.clone()
        aatype_x_t = aatype_target.clone()
        struct_loss_mask = torch.zeros_like(struct_target, dtype=torch.bool)
        aa_loss_mask = torch.zeros_like(aatype_target, dtype=torch.bool)

        struct_idx = (~is_ngs).nonzero(as_tuple=True)[0]
        ngs_idx = is_ngs.nonzero(as_tuple=True)[0]

        if struct_idx.numel() > 0:
            (
                struct_noised,
                aatype_noised,
                struct_single,
            ) = self._construct_x_t_with_targets(
                struct_target[struct_idx],
                aatype_target[struct_idx],
                mask_scale=mask_scale,
            )
            struct_t[struct_idx] = struct_noised["t"]
            aatype_t[struct_idx] = aatype_noised["t"]
            single_modality_index[struct_idx] = struct_single
            struct_x_t[struct_idx] = struct_noised["x_t"]
            aatype_x_t[struct_idx] = aatype_noised["x_t"]
            struct_loss_mask[struct_idx] = struct_noised["mask"]
            aa_loss_mask[struct_idx] = aatype_noised["mask"]

        if ngs_idx.numel() > 0:
            (
                struct_noised_ngs,
                aatype_noised_ngs,
                ngs_single,
            ) = self._construct_x_t_seq_only_dummy(
                struct_target[ngs_idx], aatype_target[ngs_idx]
            )
            struct_t[ngs_idx] = struct_noised_ngs["t"]
            aatype_t[ngs_idx] = aatype_noised_ngs["t"]
            single_modality_index[ngs_idx] = ngs_single
            struct_x_t[ngs_idx] = struct_noised_ngs["x_t"]
            aatype_x_t[ngs_idx] = aatype_noised_ngs["x_t"]
            struct_loss_mask[ngs_idx] = struct_noised_ngs["mask"]
            aa_loss_mask[ngs_idx] = aatype_noised_ngs["mask"]

        if self._zero_struct_loss_enabled():
            struct_loss_mask = torch.zeros_like(struct_loss_mask)

        return (
            {"t": struct_t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": aatype_t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    def _cdr_maskable_mask(self, tokens, cdr_mask_half: Optional[torch.Tensor]):
        """non_special & CDR residues (framework excluded when enabled)."""
        base = self.get_non_special_symbol_mask(tokens)
        if cdr_mask_half is None:
            return base
        if cdr_mask_half.shape != base.shape:
            # pad/truncate to token length
            out = torch.zeros_like(base)
            n = min(cdr_mask_half.size(1), base.size(1))
            out[:, :n] = cdr_mask_half[:, :n].bool()
            cdr_mask_half = out
        return base & cdr_mask_half.bool()

    def construct_x_t(
        self, struct_target, aatype_target, cdr_mask=None, mask_scale: float = 1.0
    ):
        # Antigen-conditioned Ab design: force joint denoising of aa+struct.
        if bool(getattr(getattr(self.cfg, "antigen_condition", None), "enable", False)):
            joint = float(getattr(self.cfg, "joint_loss_ratio", 0.0))
            others = (
                float(getattr(self.cfg, "single_modality_ratio", 0.0))
                + float(getattr(self.cfg, "folding_loss_ratio", 0.0))
                + float(getattr(self.cfg, "inverse_folding_loss_ratio", 0.0))
                + float(getattr(self.cfg, "independent_loss_ratio", 0.0))
            )
            if joint >= 0.999 and others <= 1e-6:
                return self._construct_x_t_joint_only(
                    struct_target,
                    aatype_target,
                    cdr_mask=cdr_mask,
                    mask_scale=mask_scale,
                )
        return self._construct_x_t_with_targets(
            struct_target,
            aatype_target,
            cdr_mask=cdr_mask,
            mask_scale=mask_scale,
        )

    def _construct_x_t_joint_only(
        self, struct_target, aatype_target, cdr_mask=None, mask_scale: float = 1.0
    ):
        """Same diffusion timestep for Ab struct and aa; both modalities supervised."""
        bsz = struct_target.size(0)
        device = struct_target.device
        t = torch.randint(
            1,
            self.cfg.num_diffusion_timesteps + 1,
            (bsz,),
            device=device,
        )
        struct_maskable = self._cdr_maskable_mask(struct_target, cdr_mask)
        aa_maskable = self._cdr_maskable_mask(aatype_target, cdr_mask)
        struct_x_t, struct_loss_mask = self.q_sample(
            struct_target,
            t,
            self.get_modality_type(struct_target),
            maskable_mask=struct_maskable,
            mask_scale=mask_scale,
        )
        aatype_x_t, aa_loss_mask = self.q_sample(
            aatype_target,
            t,
            self.get_modality_type(aatype_target),
            maskable_mask=aa_maskable,
            mask_scale=mask_scale,
        )
        # Allow aa↔struct full attention within the antibody concat.
        single_modality_index = torch.zeros(bsz, dtype=torch.bool, device=device)
        return (
            {"t": t, "x_t": struct_x_t, "mask": struct_loss_mask},
            {"t": t, "x_t": aatype_x_t, "mask": aa_loss_mask},
            single_modality_index,
        )

    def _construct_x_t_hybrid_noise(
        self,
        struct_target,
        aatype_target,
        cdr_mask,
        prog_scale: float,
        cdr_noise_ratio: float,
    ):
        """Mix original full masking (scale=1) with CDR-only progressive scale.

        ``cdr_noise_ratio`` of the batch uses CDR maskable + ``prog_scale``;
        the remainder uses full non-special maskable at scale 1.0.
        """
        bsz = struct_target.size(0)
        device = struct_target.device
        ratio = float(max(0.0, min(1.0, cdr_noise_ratio)))
        n_cdr = int(round(bsz * ratio))
        n_cdr = max(0, min(bsz, n_cdr))
        # Prefer keeping both modes when batch is large enough.
        if bsz >= 2 and 0 < ratio < 1:
            n_cdr = max(1, min(bsz - 1, n_cdr))

        perm = torch.randperm(bsz, device=device)
        cdr_idx = perm[:n_cdr]
        orig_idx = perm[n_cdr:]

        struct_t = torch.zeros(bsz, dtype=torch.long, device=device)
        aatype_t = torch.zeros(bsz, dtype=torch.long, device=device)
        struct_x_t = struct_target.clone()
        aatype_x_t = aatype_target.clone()
        struct_loss_mask = torch.zeros_like(struct_target, dtype=torch.bool)
        aa_loss_mask = torch.zeros_like(aatype_target, dtype=torch.bool)
        single_modality_index = torch.zeros(bsz, dtype=torch.bool, device=device)
        scale_vec = torch.ones(bsz, dtype=torch.float, device=device)

        def _scatter(idx, struct_n, aa_n, single, scale_val):
            if idx.numel() == 0:
                return
            struct_t[idx] = struct_n["t"]
            aatype_t[idx] = aa_n["t"]
            struct_x_t[idx] = struct_n["x_t"]
            aatype_x_t[idx] = aa_n["x_t"]
            struct_loss_mask[idx] = struct_n["mask"]
            aa_loss_mask[idx] = aa_n["mask"]
            single_modality_index[idx] = single
            scale_vec[idx] = float(scale_val)

        if cdr_idx.numel() > 0:
            cdr_half = None if cdr_mask is None else cdr_mask[cdr_idx]
            struct_n, aa_n, single = self.construct_x_t(
                struct_target[cdr_idx],
                aatype_target[cdr_idx],
                cdr_mask=cdr_half,
                mask_scale=prog_scale,
            )
            _scatter(cdr_idx, struct_n, aa_n, single, prog_scale)

        if orig_idx.numel() > 0:
            struct_n, aa_n, single = self.construct_x_t(
                struct_target[orig_idx],
                aatype_target[orig_idx],
                cdr_mask=None,
                mask_scale=1.0,
            )
            _scatter(orig_idx, struct_n, aa_n, single, 1.0)

        return (
            {
                "t": struct_t,
                "x_t": struct_x_t,
                "mask": struct_loss_mask,
                "mask_scale": scale_vec,
            },
            {
                "t": aatype_t,
                "x_t": aatype_x_t,
                "mask": aa_loss_mask,
                "mask_scale": scale_vec,
            },
            single_modality_index,
        )

    def compute_loss(self, batch, weighting="linear", global_step: int = 0):
        struct_target = batch["struct_tokens"]["targets"]
        aatype_target = batch["aatype_tokens"]["targets"]
        cdr_mask = None
        cdr_gen_on = bool(
            getattr(getattr(self.cfg, "cdr_generation", None), "enable", False)
        )
        sched_cfg = getattr(self.cfg, "mask_schedule", None)
        cdr_noise_ratio = float(getattr(sched_cfg, "cdr_noise_ratio", 0.0) or 0.0)
        # Hybrid mix needs CDR annotations even when only a fraction uses them.
        if cdr_gen_on or cdr_noise_ratio > 0:
            cdr_mask = batch.get("cdr_mask")
            if cdr_mask is not None:
                cdr_mask = cdr_mask.to(struct_target.device)

        prog_scale = self._mask_scale(global_step)
        # Default scalar scale for non-hybrid paths / logging.
        mask_scale = prog_scale

        if self._mixed_training_enabled() and "is_ngs" in batch:
            (
                struct_noised,
                aatype_noised,
                single_modality_index,
            ) = self._construct_x_t_mixed(
                struct_target,
                aatype_target,
                batch["is_ngs"].to(struct_target.device),
                mask_scale=mask_scale,
            )
            scale_vec = None
        elif cdr_noise_ratio > 0:
            if cdr_mask is None:
                raise ValueError(
                    "mask_schedule.cdr_noise_ratio>0 requires batch['cdr_mask']; "
                    "set datamodule.require_cdr=true"
                )
            (
                struct_noised,
                aatype_noised,
                single_modality_index,
            ) = self._construct_x_t_hybrid_noise(
                struct_target,
                aatype_target,
                cdr_mask=cdr_mask,
                prog_scale=prog_scale,
                cdr_noise_ratio=cdr_noise_ratio,
            )
            scale_vec = struct_noised["mask_scale"]
            mask_scale = float(scale_vec.mean().item())
        else:
            # Legacy: cdr_generation.enable → all CDR; else full sequence.
            use_cdr = cdr_mask if cdr_gen_on else None
            (
                struct_noised,
                aatype_noised,
                single_modality_index,
            ) = self.construct_x_t(
                struct_target,
                aatype_target,
                cdr_mask=use_cdr,
                mask_scale=mask_scale,
            )
            scale_vec = None

        if self._zero_struct_loss_enabled() and not self._mixed_training_enabled():
            struct_noised["mask"] = struct_noised["mask"].masked_fill(
                struct_noised["mask"], False
            )
        x_t = torch.concat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)
        antigen_context = self.encode_antigen(batch.get("antigen"))
        if self.cfg.self_mixup.enable:
            model_outputs, mixup_loss_mask = self.self_mixup(
                x_t=x_t,
                single_modality_index=single_modality_index,
            )
            (
                struct_noised["mask"],
                aatype_noised["mask"],
            ) = mixup_loss_mask.chunk(2, dim=1)
        else:
            model_outputs = self.forward(
                input_ids=x_t,
                single_modality=single_modality_index,
                antigen_hidden_states=(
                    antigen_context["hidden_states"]
                    if antigen_context is not None
                    else None
                ),
                antigen_attention_mask=(
                    antigen_context["attention_mask"]
                    if antigen_context is not None
                    else None
                ),
            )

        struct_logits, aatype_logits = model_outputs["logits"].chunk(2, dim=1)
        num_timesteps = self.cfg.num_diffusion_timesteps
        # Per-sample scale when hybrid; else scalar broadcast.
        if scale_vec is None:
            scale_for_t = struct_noised["t"].new_full(
                struct_noised["t"].shape, float(mask_scale)
            ).float()
        else:
            scale_for_t = scale_vec.float()
        # Use effective_t = t * mask_scale so loss weights match actual noise.
        struct_t_eff = (struct_noised["t"].float() * scale_for_t).clamp(min=1.0)
        aatype_t_eff = (aatype_noised["t"].float() * scale_for_t).clamp(min=1.0)
        # Keep t=0 (clean modality) as 0 so weight formula matches prior behavior.
        struct_t_eff = torch.where(
            struct_noised["t"] == 0, torch.zeros_like(struct_t_eff), struct_t_eff
        )
        aatype_t_eff = torch.where(
            aatype_noised["t"] == 0, torch.zeros_like(aatype_t_eff), aatype_t_eff
        )
        struct_weight = {
            "linear": (
                num_timesteps - (struct_t_eff - 1)
            ),
            "constant": num_timesteps * torch.ones_like(struct_t_eff),
        }[weighting][:, None].float() / num_timesteps
        struct_weight = struct_weight.expand(struct_target.size())

        aatype_weight = {
            "linear": (
                num_timesteps - (aatype_t_eff - 1)
            ),
            "constant": num_timesteps * torch.ones_like(aatype_t_eff),
        }[weighting][:, None].float() / num_timesteps
        aatype_weight = aatype_weight.expand(aatype_target.size())

        aux_outputs = {
            "mask_scale": mask_scale,
            "cdr_noise_ratio": cdr_noise_ratio,
            "prog_scale": prog_scale,
        }
        if antigen_context is not None and "antigen" in batch:
            ab_mask = x_t.ne(self.pad_id)
            epitope_logits = self.predict_epitopes(
                antigen_context.get(
                    "encoder_hidden_states", antigen_context["hidden_states"]
                ),
                antibody_hidden_states=model_outputs.get("last_hidden_state"),
                antigen_attention_mask=antigen_context.get(
                    "pad_attention_mask", antigen_context.get("attention_mask")
                ),
                antibody_attention_mask=ab_mask,
            )
            if epitope_logits is not None:
                aux_outputs["epitope_logits"] = epitope_logits
                if "epitope_labels" in batch["antigen"]:
                    aux_outputs["epitope_labels"] = batch["antigen"][
                        "epitope_labels"
                    ].to(epitope_logits.device)
                    aux_outputs["epitope_mask"] = batch["antigen"][
                        "epitope_mask"
                    ].to(epitope_logits.device)
                    aux_outputs["epitope_loss_weight"] = float(
                        self.cfg.antigen_condition.epitope_loss_weight
                    )

        return (
            {
                "aatype": aatype_logits,
                "struct": struct_logits,
            },  # model pred logits
            {
                "aatype": aatype_target,
                "struct": struct_target,
            },  # training targets
            {  # training loss mask
                "aatype": aatype_noised["mask"],
                "struct": struct_noised["mask"],
            },
            {
                "aatype": aatype_weight,
                "struct": struct_weight,
            },  # training loss weight
            aux_outputs,
        )

    def _get_backbone_esm(self):
        esm = getattr(self.net, "esm", None)
        if esm is not None:
            return esm
        base = getattr(self.net, "base_model", None)
        if base is not None:
            model = getattr(base, "model", base)
            return getattr(model, "esm", None)
        return None

    def _enable_cross_attn_grads(self):
        esm = self._get_backbone_esm()
        if esm is None:
            return
        for layer in esm.encoder.layer:
            if hasattr(layer, "conditional_crossattention"):
                for p in layer.conditional_crossattention.parameters():
                    p.requires_grad = True

    def freeze_for_stage1(self):
        for p in self.net.parameters():
            p.requires_grad = not bool(self.cfg.two_stage.freeze_backbone)
        self._enable_cross_attn_grads()
        if exists(self.epitope_head):
            for p in self.epitope_head.parameters():
                p.requires_grad = True
        if exists(self.epitope_feature_embed):
            for p in self.epitope_feature_embed.parameters():
                p.requires_grad = True
        if exists(self.antigen_encoder):
            for p in self.antigen_encoder.parameters():
                p.requires_grad = not bool(self.cfg.two_stage.freeze_antigen_encoder)
        if self.cfg.gradient_ckpt:
            self._ensure_input_require_grads()
        self._conditional_stage = "stage1"

    def unfreeze_for_stage2(self):
        # With LoRA: only adapters / modules_to_save, never full 650M base grads
        # (full unfreeze previously OOM'd ~80GB at stage1→stage2).
        if getattr(self.cfg.lora, "enable", False):
            for name, p in self.net.named_parameters():
                train = ("lora_" in name) or ("modules_to_save" in name)
                p.requires_grad = train
        else:
            for p in self.net.parameters():
                p.requires_grad = True
        self._enable_cross_attn_grads()
        if exists(self.epitope_head):
            for p in self.epitope_head.parameters():
                p.requires_grad = True
        if exists(self.epitope_feature_embed):
            for p in self.epitope_feature_embed.parameters():
                p.requires_grad = True
        if exists(self.antigen_encoder):
            # Keep antigen encoder frozen in stage2 by default for memory;
            # override via two_stage.freeze_antigen_encoder=false.
            freeze_ag = bool(
                getattr(self.cfg.two_stage, "freeze_antigen_encoder", True)
            )
            for p in self.antigen_encoder.parameters():
                p.requires_grad = not freeze_ag
        if self.cfg.gradient_ckpt:
            self._ensure_input_require_grads()
        self._conditional_stage = "stage2"

    def forward_encoder(self, input_tokens, **kwargs):
        return {}

    def initialize_output_tokens(
        self, input_tokens, partial_masks=None, **kwargs
    ):
        type_ids = self.get_modality_type(input_tokens)
        output_mask = self.get_non_special_symbol_mask(
            input_tokens, partial_masks=partial_masks
        )
        # fill the aatype part and struct part with specialized mask token
        aa_position = type_ids.eq(self.aa_type) & output_mask
        struct_position = type_ids.eq(self.struct_type) & output_mask
        output_tokens = input_tokens.masked_fill(aa_position, self.aa_mask_id)
        output_tokens = output_tokens.masked_fill(
            struct_position, self.struct_mask_id
        )
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

        return output_tokens, output_scores

    def forward_decoder(
        self,
        prev_decoder_out,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="annealing@2.2:1.0",
        antigen_context=None,
    ):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )
        net_out = self.forward(
            input_ids=output_tokens,
            antigen_hidden_states=(
                antigen_context["hidden_states"]
                if antigen_context is not None
                else None
            ),
            antigen_attention_mask=(
                antigen_context["attention_mask"]
                if antigen_context is not None
                else None
            ),
        )

        logits = net_out["logits"].log_softmax(dim=-1)
        attentions = net_out["attentions"] if need_attn_weights else None

        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        type_ids = self.get_modality_type(output_tokens)
        aa_position = type_ids.eq(self.aa_type) & output_masks
        struct_position = type_ids.eq(self.struct_type) & output_masks
        indices_aa = torch.where(aa_position)
        indices_struct = torch.where(struct_position)

        # HACK: all amino acid token id < 33, while all struct token id >= 33
        logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
        logits[indices_struct[0], indices_struct[1], :33] = -math.inf

        logits[..., self.special_token_list] = -math.inf

        logits = top_k_top_p_filtering(logits, top_p=0.95)

        if sampling_strategy == "argmax":
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            noise_scale = temperature
            _tokens, _scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )
            _tokens.masked_scatter_(
                ~output_masks, output_tokens[~output_masks]
            )
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(
                float, sampling_strategy.split("@")[1].split(":")
            )
            rate = 1 - step / max_step
            temperature = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )
        else:
            _tokens, _scores = sample_from_categorical(
                logits, temperature=temperature
            )

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=net_out["last_hidden_state"],
        )

    def get_non_special_symbol_mask(self, output_tokens, partial_masks=None):
        non_special_symbol_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.aa_bos_id)
            & output_tokens.ne(self.aa_eos_id)
            & output_tokens.ne(self.struct_bos_id)
            & output_tokens.ne(self.struct_eos_id)
        )
        if partial_masks is not None:
            non_special_symbol_mask &= ~partial_masks
        return non_special_symbol_mask

    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        type_ids,
        non_special_sym_mask,
        t,
        max_step,
    ):
        def _reparam_process(
            output_tokens,
            output_scores,
            cur_tokens,
            cur_scores,
            xt_neq_x0,
            noise,
            non_special_sym_mask,
        ):
            """This function is used to perform reparameterized decoding.

            output_tokens: [B, N]
            output_scores: [B, N]
            cur_tokens: [B, N]
            cur_scores: [B, N]
            xt_neq_x0: equivalent to not_b_t [B, N]
            non_special_sym_mask: [B, N]
            noise: either [B, N] or scalar (if using the mask noise)
            """

            # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
            _, condition, topk_mode, schedule = decoding_strategy.split("-")

            # first set the denoising rate according to the schedule
            if schedule == "linear":
                rate = 1 - t / max_step
            elif schedule == "cosine":
                rate = np.cos(t / max_step * np.pi * 0.5)
            else:
                raise NotImplementedError

            # compute the cutoff length for denoising top-k positions
            cutoff_len = (
                non_special_sym_mask.sum(1, keepdim=True).type_as(
                    output_scores
                )
                * rate
            ).long()
            # set the scores of special symbols to a large value so that they will never be selected
            _scores_for_topk = cur_scores.masked_fill(
                ~non_special_sym_mask, 1000.0
            )

            # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
            if topk_mode.startswith("stochastic"):
                noise_scale = float(topk_mode.replace("stochastic", ""))
                lowest_k_mask = topk_masking(
                    _scores_for_topk,
                    cutoff_len,
                    stochastic=True,
                    temp=noise_scale * rate,
                )
            elif topk_mode == "deterministic":
                lowest_k_mask = topk_masking(
                    _scores_for_topk, cutoff_len, stochastic=False
                )

            elif topk_mode == "positionprior":
                lowest_k_mask_1 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask_2 = topk_masking_prior(
                    _scores_for_topk, cutoff_len, stochastic=False
                )
                lowest_k_mask = lowest_k_mask_1 | lowest_k_mask_2
            else:
                raise NotImplementedError

            # Various choices to generate v_t := [v1_t, v2_t].
            # Note that
            #   v1_t governs the outcomes of tokens where b_t = 1,
            #   v2_t governs the outcomes of tokens where b_t = 0.

            # #### the `uncond` mode ####
            # In our reparameterized decoding,
            # both v1_t and v2_t can be fully determined by the current token scores .

            # #### the `cond` mode ####
            # However, we can also impose some conditional constraints on v1_t so that
            # the decoding can be performed in a more conservative manner.
            # For example, we can set v1_t = 0 only when
            # (the newly output tokens are the same as previous denoised results, AND
            # the current token score becomes lower, AND
            # the current token score is not in the top-k share among all tokens).
            if condition == "cond":
                not_v1_t = (
                    (cur_tokens == output_tokens)
                    & (cur_scores < output_scores)
                    & lowest_k_mask
                )
            elif condition == "uncond":
                not_v1_t = lowest_k_mask
            else:
                raise NotImplementedError

            # for b_t = 0, the token is set to noise if it is in the lowest k scores.
            not_v2_t = lowest_k_mask

            last_mask_position = xt_neq_x0

            masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
            if isinstance(noise, torch.Tensor):
                output_tokens.masked_scatter_(
                    masked_to_noise, noise[masked_to_noise]
                )
            elif isinstance(noise, (int, float)):
                output_tokens.masked_fill_(masked_to_noise, noise)
            else:
                raise NotImplementedError(
                    "noise should be either a tensor or a scalar"
                )
            output_scores.masked_fill_(masked_to_noise, -math.inf)

            masked_to_x0 = xt_neq_x0 & ~not_v2_t
            output_tokens.masked_scatter_(
                masked_to_x0, cur_tokens[masked_to_x0]
            )
            output_scores.masked_scatter_(
                masked_to_x0, cur_scores[masked_to_x0]
            )
            assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
            # b_{t} = (b_{t+1} & u_t) | v_t
            # For convenience, save the NOT of b_t for the next iteration
            # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
            #
            # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t (?)
            new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
            assert (new_xt_neq_x0 == not_v2_t).all()
            return new_xt_neq_x0, output_tokens, output_scores

        aa_position = type_ids.eq(self.aa_type) & non_special_sym_mask
        struct_position = type_ids.eq(self.struct_type) & non_special_sym_mask
        new_xt_neq_x0 = xt_neq_x0.clone()
        new_xt_neq_x0_aa = new_xt_neq_x0.fill_(False)
        new_xt_neq_x0_struct = new_xt_neq_x0.fill_(False)
        if aa_position.any():
            new_xt_neq_x0_aa, output_tokens, output_scores = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & aa_position,
                noise=self.aa_mask_id,
                non_special_sym_mask=aa_position,
            )
        if struct_position.any():
            (
                new_xt_neq_x0_struct,
                output_tokens,
                output_scores,
            ) = _reparam_process(
                output_tokens=output_tokens,
                output_scores=output_scores,
                cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                xt_neq_x0=xt_neq_x0 & struct_position,
                noise=self.struct_mask_id,
                non_special_sym_mask=struct_position,
            )
        new_xt_neq_x0 = new_xt_neq_x0_aa | new_xt_neq_x0_struct
        return new_xt_neq_x0, output_tokens, output_scores

    def generate(
        self,
        input_tokens,
        max_iter=None,
        temperature=1.0,
        partial_masks=None,
        unmasking_strategy="stochastic1.0",  # [stochastic{temperature}, deterministic]
        sampling_strategy="annealing@2.0:0.1",
        antigen_batch=None,
    ):
        self.eval()
        max_iter = max_iter
        temperature = temperature
        antigen_context = self.encode_antigen(antigen_batch)

        # 0) encoding
        encoder_out = self.forward_encoder(input_tokens)
        # 1) initialized from all mask tokens
        (
            initial_output_tokens,
            initial_output_scores,
        ) = self.initialize_output_tokens(
            input_tokens, encoder_out=encoder_out, partial_masks=partial_masks
        )
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            type_ids=self.get_modality_type(initial_output_tokens),
        )

        prev_decoder_out["output_masks"] = self.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        for step in tqdm(range(max_iter), desc="Decoding"):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy,
                    antigen_context=antigen_context,
                )

            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )

            (
                output_masks,
                result_tokens,
                result_scores,
            ) = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                type_ids=prev_decoder_out["type_ids"].clone(),
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
            )

            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out["history"],
            )

        decoder_out = prev_decoder_out
        return {
            "output_tokens": decoder_out["output_tokens"],
        }
