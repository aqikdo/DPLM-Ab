# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, List, Union

import torch
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric

from byprot import utils
from byprot.tasks import TaskLitModule, register_task
from byprot.utils.config import compose_config as Cfg
from byprot.utils.config import merge_config

log = utils.get_logger(__name__)


def cal_index_acc(logits, target, loss_mask, bit_level=False):
    if not bit_level:
        model_pred = logits.argmax(dim=-1)
        index_match = (model_pred == target) & loss_mask
        index_accuracy = index_match.sum() / loss_mask.sum()
        return index_accuracy
    else:
        model_pred = logits.argmax(dim=-1)
        label_mask_expand = loss_mask[..., None].expand(
            model_pred.shape
        )  # B x L x 13
        total_bits = label_mask_expand.sum()
        bitwise_match = (model_pred == target) & label_mask_expand
        bitwise_accuracy = bitwise_match.sum() / total_bits
        index_accuracy = (
            bitwise_match.sum(dim=-1) == bitwise_match.shape[-1]
        ).sum() / loss_mask.sum()
        return index_accuracy, bitwise_accuracy


@register_task("lm/dplm2")
class DPLM2TrainingTask(TaskLitModule):
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            watch_t1_t2_loss=False,
            cal_constant_loss=False,
            weight="constant",
        ),
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        self.build_model()
        self.tokenizer = self.model.tokenizer

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == "fit":
            log.info(f"\n{self.model}")
        elif self.stage == "test":
            self.test_step_outputs = []

    def on_train_start(self):
        super().on_train_start()
        cfg = getattr(self.model, "cfg", None)
        if (
            cfg is not None
            and getattr(getattr(cfg, "antigen_condition", None), "enable", False)
            and getattr(getattr(cfg, "two_stage", None), "enable", False)
        ):
            if self.global_step < int(cfg.two_stage.stage1_steps):
                self.model.freeze_for_stage1()
            else:
                self.model.unfreeze_for_stage2()

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        cfg = getattr(self.model, "cfg", None)
        if not (
            cfg is not None
            and getattr(getattr(cfg, "antigen_condition", None), "enable", False)
            and getattr(getattr(cfg, "two_stage", None), "enable", False)
        ):
            return
        if (
            self.global_step >= int(cfg.two_stage.stage1_steps)
            and getattr(self.model, "_conditional_stage", None) != "stage2"
        ):
            self.model.unfreeze_for_stage2()
            scale = float(cfg.two_stage.stage2_lr_scale)
            for optimizer in self.trainer.optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["lr"] * scale

    def on_before_optimizer_step(self, optimizer):
        if self.global_rank == 0:
            grad_norm_dict = grad_norm(
                self.trainer.strategy.model, norm_type=2
            )
            self.log_dict(grad_norm_dict)

    def configure_optimizers(self):
        """Param groups: LoRA / other, cross-attn (scaled), antigen encoder (scaled)."""
        from byprot.utils.lr_scheduler import get_scheduler
        from byprot.utils.optim import get_optimizer

        opt_cfg = self.hparams.optimizer
        base_lr = float(opt_cfg.lr)
        model_cfg = getattr(self.model, "cfg", None)
        xattn_scale = 1.0
        ag_scale = 1.0
        if model_cfg is not None and getattr(
            getattr(model_cfg, "antigen_condition", None), "enable", False
        ):
            xattn_scale = float(
                getattr(model_cfg.antigen_condition, "cross_attn_lr_scale", 1.0)
            )
            ag_scale = float(
                getattr(
                    getattr(model_cfg, "two_stage", None),
                    "antigen_encoder_lr_scale",
                    1.0,
                )
            )

        # Ensure stage freeze flags match intended trainable set before grouping.
        if (
            model_cfg is not None
            and getattr(getattr(model_cfg, "antigen_condition", None), "enable", False)
            and getattr(getattr(model_cfg, "two_stage", None), "enable", False)
            and int(model_cfg.two_stage.stage1_steps) <= 0
        ):
            self.model.unfreeze_for_stage2()

        cross, ag, other = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "conditional_crossattention" in name:
                cross.append(p)
            elif "antigen_encoder" in name:
                ag.append(p)
            else:
                other.append(p)

        param_groups = []
        if other:
            param_groups.append(
                {"params": other, "lr": base_lr, "name": "lora_other"}
            )
        if cross:
            param_groups.append(
                {
                    "params": cross,
                    "lr": base_lr * xattn_scale,
                    "name": "cross_attn",
                }
            )
        if ag:
            param_groups.append(
                {
                    "params": ag,
                    "lr": base_lr * ag_scale,
                    "name": "antigen_encoder",
                }
            )
        if not param_groups:
            param_groups = [p for p in self.parameters() if p.requires_grad]

        log.info(
            "Optimizer groups: "
            f"lora_other={len(other)}@{base_lr:g}, "
            f"cross_attn={len(cross)}@{(base_lr * xattn_scale):g}, "
            f"antigen_encoder={len(ag)}@{(base_lr * ag_scale):g}"
        )
        optimizer = get_optimizer(opt_cfg, param_groups)
        if (
            "lr_scheduler" in self.hparams
            and self.hparams.lr_scheduler is not None
        ):
            lr_scheduler, extra_kwargs = get_scheduler(
                self.hparams.lr_scheduler, optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, **extra_kwargs},
            }
        return optimizer

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(
            cfg=self.hparams.model, group="model"
        )

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(
            cfg=self.hparams.criterion
        )
        self.criterion.ignore_index = self.tokenizer.pad_token_id

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()

        # Multi-modal valid loss
        self.eval_struct_loss = MeanMetric()
        self.eval_aatype_loss = MeanMetric()
        self.eval_struct_acc = MeanMetric()
        self.eval_aatype_acc = MeanMetric()
        self.eval_epitope_loss = MeanMetric()
        self.eval_epitope_acc = MeanMetric()

    def load_from_ckpt(self, ckpt_path, not_load=False):
        # do not load state dict from ckpt, just use the initialized parameters.
        if not_load:
            return
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        del state_dict
        print(
            f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def step(self, batch):
        """batch is a Dict containing:

        - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
        - corrd_mask: BooltTensor [bsz, len], where valid coordinates
            are set True, otherwise False
        - lengths: int [bsz, len], protein sequence lengths
        - tokens: LongTensor [bsz, len], sequence of amino acids
        """
        weighting = self.hparams.learning.weight
        outputs = self.model.compute_loss(
            batch, weighting=weighting, global_step=int(self.global_step)
        )
        if len(outputs) == 4:
            logits, targets, loss_masks, weights = outputs
            aux_outputs = {}
        else:
            logits, targets, loss_masks, weights, aux_outputs = outputs

        loss, logging_output = self.criterion(
            logits,
            targets,
            loss_masks,
            weights,
            watch_t1_t2_loss=self.hparams.learning.watch_t1_t2_loss,
            cal_constant_loss=self.hparams.learning.cal_constant_loss,
        )

        if "mask_scale" in aux_outputs:
            logging_output["mask_scale"] = float(aux_outputs["mask_scale"])
        if "prog_scale" in aux_outputs:
            logging_output["prog_scale"] = float(aux_outputs["prog_scale"])
        if "cdr_noise_ratio" in aux_outputs:
            logging_output["cdr_noise_ratio"] = float(aux_outputs["cdr_noise_ratio"])

        if "epitope_logits" in aux_outputs and "epitope_labels" in aux_outputs:
            epi_w = float(aux_outputs.get("epitope_loss_weight", 1.0))
            if epi_w > 0.0:
                epitope_logits = aux_outputs["epitope_logits"]
                epitope_labels = aux_outputs["epitope_labels"]
                epitope_mask = aux_outputs["epitope_mask"].bool()
                valid_logits = epitope_logits[epitope_mask]
                valid_labels = epitope_labels[epitope_mask]
                if valid_logits.numel() > 0:
                    pos = valid_labels.sum()
                    neg = valid_labels.numel() - pos
                    pos_weight = (
                        (neg / pos.clamp_min(1.0))
                        if pos.item() > 0
                        else valid_labels.new_tensor(1.0)
                    )
                    epitope_loss = F.binary_cross_entropy_with_logits(
                        valid_logits,
                        valid_labels,
                        pos_weight=pos_weight,
                    )
                    loss = loss + epi_w * epitope_loss
                    with torch.no_grad():
                        epitope_pred = (valid_logits > 0).float()
                        epitope_acc = (epitope_pred == valid_labels).float().mean()
                    logging_output["epitope/loss"] = epitope_loss.detach()
                    logging_output["epitope/acc"] = epitope_acc.detach()
                    logging_output["epitope/pos_weight"] = pos_weight.detach()
                    logging_output["epitope/sample_size"] = valid_labels.numel()

        # calculate index accuracy
        logging_output["aatype/index_accuracy"] = cal_index_acc(
            logits["aatype"], targets["aatype"], loss_masks["aatype"]
        )
        if len(loss_masks["struct"].shape) == (
            len(targets["struct"].shape) - 1
        ):
            # if bit-based modeling,
            # the loss is in B x L x 13 and label_mask is in B x L
            (
                logging_output["struct/index_accuracy"],
                logging_output["struct/bit_accuracy"],
            ) = cal_index_acc(
                logits["struct"],
                targets["struct"],
                loss_masks["struct"],
                bit_level=True,
            )
        else:
            logging_output["struct/index_accuracy"] = cal_index_acc(
                logits["struct"], targets["struct"], loss_masks["struct"]
            )

        if torch.isnan(loss):
            print("Loss NAN on step ", self.global_step)
            loss = loss * 0
            logging_output["nll_loss"] = logging_output["nll_loss"] * 0
            logging_output["fullseq_loss"] = logging_output["fullseq_loss"] * 0
            logging_output["fullseq_nll_loss"] = (
                logging_output["fullseq_nll_loss"] * 0
            )
            logging_output["ppl"] = logging_output["ppl"] * 0

        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log train metrics
        self.log(
            "global_step",
            self.global_step,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("lr", self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(
                f"train/{log_key}",
                log_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return {"loss": loss}

    # -------# Evaluating #-------- #
    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log other metrics
        sample_size = logging_output["sample_size"]
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(
            logging_output["nll_loss"], weight=sample_size
        )

        for log_key in logging_output:
            if "constant_diff_loss" not in log_key:
                continue
            log_value = logging_output[log_key]
            eval_type = log_key.split("/")[0]
            if eval_type == "aatype":
                self.eval_aatype_loss.update(log_value, weight=sample_size)
            elif eval_type == "struct":
                self.eval_struct_loss.update(log_value, weight=sample_size)
            else:
                raise NotImplementedError
        self.eval_aatype_acc.update(
            logging_output["aatype/index_accuracy"], weight=sample_size
        )
        self.eval_struct_acc.update(
            logging_output["struct/index_accuracy"], weight=sample_size
        )
        if "epitope/loss" in logging_output:
            epi_size = int(logging_output.get("epitope/sample_size", 1))
            self.eval_epitope_loss.update(logging_output["epitope/loss"], weight=epi_size)
            self.eval_epitope_acc.update(logging_output["epitope/acc"], weight=epi_size)

        return {"loss": loss}

    def on_validation_epoch_end(self):
        log_key = "test" if self.stage == "test" else "val"

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        eval_aatype_loss = self.eval_aatype_loss.compute()
        self.eval_aatype_loss.reset()
        eval_struct_loss = self.eval_struct_loss.compute()
        self.eval_struct_loss.reset()
        eval_aatype_accuracy = self.eval_aatype_acc.compute()
        self.eval_aatype_acc.reset()
        eval_struct_accuracy = self.eval_struct_acc.compute()
        self.eval_struct_acc.reset()
        eval_epitope_loss = self.eval_epitope_loss.compute()
        self.eval_epitope_loss.reset()
        eval_epitope_accuracy = self.eval_epitope_acc.compute()
        self.eval_epitope_acc.reset()

        self.log(
            f"{log_key}/loss",
            eval_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/nll_loss",
            eval_nll_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/ppl",
            eval_ppl,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            f"{log_key}/aatype_loss",
            eval_aatype_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/struct_loss",
            eval_struct_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/aatype_index_accuracy",
            eval_aatype_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{log_key}/struct_index_accuracy",
            eval_struct_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if not torch.isnan(eval_epitope_loss):
            self.log(
                f"{log_key}/epitope_loss",
                eval_epitope_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{log_key}/epitope_acc",
                eval_epitope_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        if self.stage == "fit":
            self.val_ppl_best.update(eval_ppl)
            self.log(
                "val/ppl_best",
                self.val_ppl_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )

        super().on_validation_epoch_end()
