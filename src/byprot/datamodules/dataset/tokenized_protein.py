import math
import os
from typing import Iterable, Optional, Sequence, TypeVar

import datasets
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from transformers import EsmTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken

from byprot import utils

log = utils.get_logger(__name__)
T_co = TypeVar("T_co", covariant=True)


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]


def preprocess_dataset(csv_path, data_bin, split):
    def remove_lowconf_ends(row, threshold=50):
        aa_seq, ss_seq, plddt = (
            row["aa_seq"],
            row["struct_seq"],
            np.array(row["plddt"]),
        )
        ss_seq = ss_seq.split(",")
        modeled_idx = np.where(plddt > threshold)[0]
        min_modeled_idx = np.min(modeled_idx)
        max_modeled_idx = np.max(modeled_idx)
        aa_seq = aa_seq[min_modeled_idx : (max_modeled_idx + 1)]
        ss_seq = ss_seq[min_modeled_idx : (max_modeled_idx + 1)]
        plddt = plddt[min_modeled_idx : (max_modeled_idx + 1)]
        ss_seq = ",".join(ss_seq)
        row["aa_seq"], row["struct_seq"], row["plddt"] = aa_seq, ss_seq, plddt
        return row

    # preprocess dataset
    afdb_pdb = pd.read_csv(csv_path)

    afdb_pdb.dropna(subset=["aa_seq"], inplace=True)
    afdb = afdb_pdb[afdb_pdb["split"] == "afdb_swissprot"]
    pdb = afdb_pdb[afdb_pdb["split"] == "pdb"]

    afdb["plddt"] = afdb["plddt"].apply(
        lambda l: [float(a) for a in l.split(",") if len(a) > 0]
    )
    afdb = afdb.apply(
        lambda row: remove_lowconf_ends(row, threshold=70), axis=1
    )
    pdb["plddt"] = pdb["plddt"].apply(
        lambda l: [float(a) for a in l.split(",") if len(a) > 0]
    )
    pdb = pdb.apply(lambda row: remove_lowconf_ends(row, threshold=70), axis=1)

    afdb["plddt_std"] = afdb["plddt"].apply(lambda l: np.std(l))
    afdb = afdb[afdb["plddt_std"] < 15]
    remaining_set = pd.concat([afdb, pdb], axis=0)

    remaining_set = remaining_set[remaining_set["aa_seq"].str.len() <= 1024]
    remaining_set = remaining_set[
        (remaining_set["split"] == "pdb")
        | (
            (remaining_set["avg_plddt"].notna())
            & (remaining_set["avg_plddt"] > 85)
        )
    ]
    remaining_set["cluster"] = remaining_set["cluster"].apply(lambda x: str(x))

    # save to huggingface dataset
    valid_set = afdb_pdb[afdb_pdb["split"] == "cameo2022"]
    valid_set = datasets.Dataset.from_pandas(valid_set)
    training_set = datasets.Dataset.from_pandas(remaining_set)

    def add_seqlen(example):
        example["length"] = len(example["aa_seq"])
        return example

    training_set = training_set.map(add_seqlen)
    valid_set = valid_set.map(add_seqlen)

    os.makedirs(data_bin, exist_ok=True)
    training_set.save_to_disk(os.path.join(data_bin, "train"), num_proc=1)
    valid_set.save_to_disk(os.path.join(data_bin, "valid"), num_proc=1)

    log.info(f"Preprocessed dataset from {csv_path}.")
    return training_set if split == "train" else valid_set


class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close
    together."""

    def __init__(
        self,
        sequence_lengths: Iterable,
        bucket_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        epoch: int = 0,
    ):
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(
            math.ceil(len(self.data) * 1.0 / self.num_replicas)
        )
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size]
            for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = epoch
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class MappedSortishSampler(Sampler):
    """SortishSampler over a subset of dataset indices (yields original idxs).

    Used when mixed_vhh + struct_sample_ratio<=0 so pure-NGS training matches
    the dummy_struct recipe (length-bucketed Sortish), not StructRatio streaming.
    """

    def __init__(
        self,
        indices,
        sequence_lengths,
        bucket_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        epoch: int = 0,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        if len(self.indices) == 0:
            raise ValueError("MappedSortishSampler requires non-empty indices")
        subset_lens = np.asarray(sequence_lengths)[self.indices]
        self._sampler = SortishSampler(
            subset_lens,
            bucket_size,
            num_replicas=num_replicas,
            rank=rank,
            epoch=epoch,
        )

    def __iter__(self):
        for local_idx in self._sampler:
            yield int(self.indices[local_idx])

    def __len__(self):
        return len(self._sampler)

    @property
    def num_replicas(self):
        return self._sampler.num_replicas

    @property
    def epoch(self):
        return self._sampler.epoch

    @epoch.setter
    def epoch(self, value):
        self._sampler.epoch = value

    def set_epoch(self, epoch):
        self._sampler.epoch = epoch


class ApproxBatchSampler(BatchSampler):
    """
    Parameters:
    -----------
    sampler : Pytorch Sampler
            Choose base sampler class to use for bucketing

    max_tokens : int
            Maximum number of tokens per batch

    max_batch: int
            Maximum batch size

    sample_lengths : array-like
            List of lengths of sequences in the order of the dataset
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        msa_depth=None,
        drop_last=False,
        batch_size=None,
        max_len=512,
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.batches = self._build_batches()

    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if (
                linear <= self.max_tokens
                and quadratic < self.max_square_tokens
            ):
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print("Current batch is empty! idx is ", idx)
                    continue
                batches.append(batch)
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
        if len(batch) > 0:
            batches.append(batch)

        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = num_samples // len(batches)
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
        return batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


class TokenizedProteinDataset(Dataset):
    """Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        csv_file: str,
        max_len=2048,
        antigen_max_len: Optional[int] = None,
        vocab_file="airkingbd/dplm2_650m",
        struct_vocab_size=8192,
    ):
        self.data_dir = data_dir
        self.split = split
        csv_path = os.path.join(self.data_dir, csv_file)
        data_path = os.path.join(self.data_dir, csv_file.replace(".csv", ""))
        try:
            self.data = load_dataset_from_hf(data_path, split)
        except:
            self.data = preprocess_dataset(csv_path, data_path, self.split)
        log.info(f"Dataset size: {len(self.data)}")

        self.max_len = max_len
        self.antigen_max_len = (
            int(antigen_max_len) if antigen_max_len is not None else int(max_len)
        )
        self.tokenizer = DPLM2Tokenizer.from_pretrained(vocab_file)

    def __len__(self):
        return len(self.data)

    def get_metadata_lens(self):
        return self.data["length"]

    def _crop_seq(self, seq: str, max_len: int):
        if len(seq) - max_len > 0:
            start = np.random.choice(len(seq) - max_len)
            stop = start + max_len
            return seq[start:stop], start, stop
        return seq, 0, len(seq)

    def _crop_antigen_with_epitope(
        self, seq: str, max_len: int, epitope_vals: Optional[Sequence[int]]
    ):
        """Prefer a window covering the densest epitope residues."""
        if len(seq) <= max_len:
            return seq, 0, len(seq)
        n = len(seq)
        if epitope_vals is None or len(epitope_vals) != n:
            return self._crop_seq(seq, max_len)
        labels = np.asarray(epitope_vals, dtype=np.int32)
        window_sums = np.convolve(labels, np.ones(max_len, dtype=np.int32), mode="valid")
        best = int(window_sums.max())
        candidates = np.flatnonzero(window_sums == best)
        start = int(np.random.choice(candidates))
        stop = start + max_len
        return seq[start:stop], start, stop

    def _wrap_aa_tokens(self, seq: str) -> str:
        return self.tokenizer.aa_cls_token + seq + self.tokenizer.aa_eos_token

    def _wrap_struct_tokens(self, struct_tokens: Sequence[str]) -> str:
        return (
            self.tokenizer.struct_cls_token
            + "".join(struct_tokens)
            + self.tokenizer.struct_eos_token
        )

    def _parse_epitope_mask(self, raw_mask, start: int, stop: int):
        if raw_mask is None or (isinstance(raw_mask, float) and np.isnan(raw_mask)):
            return None
        if isinstance(raw_mask, str):
            if "," in raw_mask:
                vals = [int(x) for x in raw_mask.split(",") if len(x) > 0]
            else:
                vals = [int(ch) for ch in raw_mask.strip() if ch in {"0", "1"}]
        elif isinstance(raw_mask, (list, tuple, np.ndarray)):
            vals = [int(x) for x in raw_mask]
        else:
            raise TypeError(f"Unsupported epitope mask type: {type(raw_mask)}")
        vals = vals[start:stop]
        return [0] + vals + [0]

    def __getitem__(self, idx):
        row = self.data[int(idx)]
        max_len = min(self.max_len, row["length"])

        # Support both mixed HF (`aa_seq`) and legacy NGS tokenized (`seq`).
        aa_raw = row.get("aa_seq") or row.get("seq")
        if not aa_raw:
            raise KeyError(
                f"Sample {idx} missing aa_seq/seq; columns={list(row.keys())}"
            )
        aatype_tokens, start, stop = self._crop_seq(aa_raw, max_len)
        aatype_tokens = self._wrap_aa_tokens(aatype_tokens)

        return_dict = {
            "aatype_tokens": aatype_tokens,
            "length": min(max_len, len(aa_raw)) + 2,
        }

        data_source = row.get("data_source")
        if data_source == "ngs" or not row.get("struct_seq"):
            return_dict["data_source"] = "ngs"
        else:
            struct_tokens = row["struct_seq"].split(",")
            if len(struct_tokens) - max_len > 0:
                struct_tokens = struct_tokens[start:stop]
            return_dict["struct_tokens"] = self._wrap_struct_tokens(struct_tokens)
            return_dict["data_source"] = "struct"

        if row.get("pdb_name"):
            return_dict["pdb_name"] = row["pdb_name"]
        if row.get("chain_id"):
            return_dict["chain_id"] = row["chain_id"]

        antigen_aa = (
            row.get("antigen_aa_seq")
            or row.get("antigen_seq")
            or row.get("ag_aa_seq")
        )
        antigen_struct = row.get("antigen_struct_seq") or row.get("ag_struct_seq")
        if antigen_aa:
            raw_epitope = row.get("epitope_mask")
            epitope_vals = None
            if raw_epitope is not None and not (
                isinstance(raw_epitope, float) and np.isnan(raw_epitope)
            ):
                if isinstance(raw_epitope, str):
                    if "," in raw_epitope:
                        epitope_vals = [
                            int(x) for x in raw_epitope.split(",") if len(x) > 0
                        ]
                    else:
                        epitope_vals = [
                            int(ch) for ch in raw_epitope.strip() if ch in {"0", "1"}
                        ]
                elif isinstance(raw_epitope, (list, tuple, np.ndarray)):
                    epitope_vals = [int(x) for x in raw_epitope]
            antigen_max_len = min(self.antigen_max_len, len(antigen_aa))
            antigen_aa, ag_start, ag_stop = self._crop_antigen_with_epitope(
                antigen_aa, antigen_max_len, epitope_vals
            )
            return_dict["has_antigen"] = True
            return_dict["antigen_aatype_tokens"] = self._wrap_aa_tokens(antigen_aa)
            return_dict["antigen_length"] = len(antigen_aa) + 2
            if antigen_struct:
                antigen_struct_tokens = antigen_struct.split(",")
                if len(antigen_struct_tokens) - antigen_max_len > 0:
                    antigen_struct_tokens = antigen_struct_tokens[ag_start:ag_stop]
                return_dict["antigen_struct_tokens"] = self._wrap_struct_tokens(
                    antigen_struct_tokens
                )
            epitope_mask = self._parse_epitope_mask(
                row.get("epitope_mask"), ag_start, ag_stop
            )
            if epitope_mask is not None:
                return_dict["epitope_labels"] = epitope_mask
            if row.get("antigen_chain"):
                return_dict["antigen_chain"] = row["antigen_chain"]
            if row.get("antigen_chain_map"):
                return_dict["antigen_chain_map"] = row["antigen_chain_map"]
        else:
            return_dict["has_antigen"] = False

        return return_dict


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DPLM2Tokenizer(EsmTokenizer):
    SPECIAL_TOKENS_ATTRIBUTES = [
        "aa_cls_token",
        "aa_eos_token",
        "aa_unk_token",
        "aa_mask_token",
        "struct_cls_token",
        "struct_eos_token",
        "struct_unk_token",
        "struct_mask_token",
        "pad_token",
    ]

    def __init__(
        self,
        vocab_file,
        aa_cls_token="<cls_aa>",
        aa_eos_token="<eos_aa>",
        aa_unk_token="<unk_aa>",
        aa_mask_token="<mask_aa>",
        struct_cls_token="<cls_struct>",
        struct_eos_token="<eos_struct>",
        struct_unk_token="<unk_struct>",
        struct_mask_token="<mask_struct>",
        pad_token="<pad>",
        **kwargs,
    ):
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {
            tok: ind for ind, tok in enumerate(self.all_tokens)
        }

        self._aa_cls_token = None
        self._aa_eos_token = None
        self._aa_unk_token = None
        self._aa_mask_token = None
        self._struct_cls_token = None
        self._struct_eos_token = None
        self._struct_unk_token = None
        self._struct_mask_token = None
        self._pad_token = None

        PreTrainedTokenizer.__init__(
            self,
            aa_cls_token=aa_cls_token,
            aa_eos_token=aa_eos_token,
            aa_unk_token=aa_unk_token,
            aa_mask_token=aa_mask_token,
            struct_cls_token=struct_cls_token,
            struct_eos_token=struct_eos_token,
            struct_unk_token=struct_unk_token,
            struct_mask_token=struct_mask_token,
            pad_token=pad_token,
            **kwargs,
        )

        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    @property
    def aa_eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_eos_token is None:
            if self.verbose:
                log.error("Using aa_eos_token, but it is not set yet.")
            return None
        return str(self._aa_eos_token)

    @property
    def aa_cls_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_cls_token is None:
            if self.verbose:
                log.error("Using aa_cls_token, but it is not set yet.")
            return None
        return str(self._aa_cls_token)

    @property
    def aa_unk_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_unk_token is None:
            if self.verbose:
                log.error("Using aa_unk_token, but it is not set yet.")
            return None
        return str(self._aa_unk_token)

    @property
    def aa_mask_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._aa_mask_token is None:
            if self.verbose:
                log.error("Using aa_mask_token, but it is not set yet.")
            return None
        return str(self._aa_mask_token)

    @property
    def struct_eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_eos_token is None:
            if self.verbose:
                log.error("Using struct_eos_token, but it is not set yet.")
            return None
        return str(self._struct_eos_token)

    @property
    def struct_cls_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_cls_token is None:
            if self.verbose:
                log.error("Using struct_cls_token, but it is not set yet.")
            return None
        return str(self._struct_cls_token)

    @property
    def struct_unk_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_unk_token is None:
            if self.verbose:
                log.error("Using struct_unk_token, but it is not set yet.")
            return None
        return str(self._struct_unk_token)

    @property
    def struct_mask_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._struct_mask_token is None:
            if self.verbose:
                log.error("Using struct_mask_token, but it is not set yet.")
            return None
        return str(self._struct_mask_token)

    @aa_cls_token.setter
    def aa_cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_cls_token"
            )
        self._aa_cls_token = value

    @aa_eos_token.setter
    def aa_eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_eos_token"
            )
        self._aa_eos_token = value

    @aa_unk_token.setter
    def aa_unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_unk_token"
            )
        self._aa_unk_token = value

    @aa_mask_token.setter
    def aa_mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the aa_mask_token"
            )
        self._aa_mask_token = value

    @struct_cls_token.setter
    def struct_cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_cls_token"
            )
        self._struct_cls_token = value

    @struct_eos_token.setter
    def struct_eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_eos_token"
            )
        self._struct_eos_token = value

    @struct_unk_token.setter
    def struct_unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_unk_token"
            )
        self._struct_unk_token = value

    @struct_mask_token.setter
    def struct_mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError(
                "Cannot set a non-string value as the struct_mask_token"
            )
        self._struct_mask_token = value


class DPLM2Collater(object):
    def __init__(self, tokenizer):
        self.tokenizer = (
            tokenizer  # DPLM2Tokenizer.from_pretrained(vocab_file)
        )

    def _dummy_struct_for_aa(self, aatype_tokens: str) -> str:
        core = aatype_tokens
        if core.startswith(self.tokenizer.aa_cls_token):
            core = core[len(self.tokenizer.aa_cls_token) :]
        if core.endswith(self.tokenizer.aa_eos_token):
            core = core[: -len(self.tokenizer.aa_eos_token)]
        unk = self.tokenizer.struct_unk_token
        return (
            self.tokenizer.struct_cls_token
            + unk * len(core)
            + self.tokenizer.struct_eos_token
        )

    def __call__(self, raw_batch):
        if len(list(zip(*raw_batch))) == 0:
            print("list idx error!")
            print(raw_batch)

        struct_tokens_list = [sample["struct_tokens"] for sample in raw_batch]

        batch_struct = self.tokenizer.batch_encode_plus(
            struct_tokens_list,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )

        batch_struct = {
            "targets": batch_struct["input_ids"],
            "attention_mask": batch_struct["attention_mask"].bool(),
        }

        aatype_list = [sample["aatype_tokens"] for sample in raw_batch]
        batch_aatype = self.tokenizer.batch_encode_plus(
            aatype_list,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        batch_aatype = {
            "targets": batch_aatype["input_ids"],
            "attention_mask": batch_aatype["attention_mask"].bool(),
        }

        batch = {
            "struct_tokens": batch_struct,
            "aatype_tokens": batch_aatype,
        }

        if any(sample.get("has_antigen") for sample in raw_batch):
            antigen_aatype_list = []
            antigen_struct_list = []
            has_antigen = []
            epitope_labels = []
            epitope_mask = []
            max_ep_len = 0

            for sample in raw_batch:
                has_ag = bool(sample.get("has_antigen"))
                has_antigen.append(has_ag)
                aa_tokens = sample.get("antigen_aatype_tokens")
                antigen_len = int(sample.get("antigen_length", 2))
                if aa_tokens is None:
                    aa_tokens = (
                        self.tokenizer.aa_cls_token + self.tokenizer.aa_eos_token
                    )
                antigen_aatype_list.append(aa_tokens)
                struct_tokens = sample.get("antigen_struct_tokens")
                if struct_tokens is None:
                    struct_tokens = self._dummy_struct_for_aa(aa_tokens)
                antigen_struct_list.append(struct_tokens)
                labels = sample.get("epitope_labels")
                if labels is None:
                    labels = [0] * antigen_len
                    valid = [0] * antigen_len
                else:
                    valid = [1] * len(labels)
                epitope_labels.append(labels)
                epitope_mask.append(valid)
                max_ep_len = max(max_ep_len, len(labels))

            batch_antigen_struct = self.tokenizer.batch_encode_plus(
                antigen_struct_list,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            batch_antigen_aatype = self.tokenizer.batch_encode_plus(
                antigen_aatype_list,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            antigen_batch = {
                "struct_tokens": {
                    "targets": batch_antigen_struct["input_ids"],
                    "attention_mask": batch_antigen_struct["attention_mask"].bool(),
                },
                "aatype_tokens": {
                    "targets": batch_antigen_aatype["input_ids"],
                    "attention_mask": batch_antigen_aatype["attention_mask"].bool(),
                },
                "has_antigen": torch.tensor(has_antigen, dtype=torch.bool),
            }
            if max_ep_len > 0:
                padded_labels = torch.zeros(
                    len(raw_batch), max_ep_len, dtype=torch.float
                )
                padded_mask = torch.zeros(
                    len(raw_batch), max_ep_len, dtype=torch.bool
                )
                for i, (labels, valid) in enumerate(zip(epitope_labels, epitope_mask)):
                    padded_labels[i, : len(labels)] = torch.tensor(
                        labels, dtype=torch.float
                    )
                    padded_mask[i, : len(valid)] = torch.tensor(valid, dtype=torch.bool)
                antigen_batch["epitope_labels"] = padded_labels
                antigen_batch["epitope_mask"] = padded_mask
            batch["antigen"] = antigen_batch

        if any("pdb_name" in sample for sample in raw_batch):
            pdb_name_list = [sample.get("pdb_name", "") for sample in raw_batch]
            batch["pdb_name"] = pdb_name_list

        return batch


class DPLM2DummyStructCollater(DPLM2Collater):
    def __call__(self, raw_batch):
        batch = []
        for sample in raw_batch:
            sample = dict(sample)
            sample["struct_tokens"] = self._dummy_struct_for_aa(sample["aatype_tokens"])
            batch.append(sample)
        return super().__call__(batch)


class MixedVHHCollater(DPLM2Collater):
    """Per-sample collater: NGS uses dummy struct; INDI2 uses real struct tokens."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._dummy = DPLM2DummyStructCollater(tokenizer)

    def __call__(self, raw_batch):
        processed = []
        is_ngs = []
        for sample in raw_batch:
            sample = dict(sample)
            if sample.get("data_source") == "ngs":
                is_ngs.append(True)
                sample["struct_tokens"] = self._dummy._dummy_struct_for_aa(
                    sample["aatype_tokens"]
                )
            else:
                is_ngs.append(False)
            processed.append(sample)
        batch = super().__call__(processed)
        batch["is_ngs"] = torch.tensor(is_ngs, dtype=torch.bool)
        return batch


def get_mixed_vhh_index_groups(ds: TokenizedProteinDataset):
    sources = np.asarray(ds.data["data_source"])
    ngs_indices = np.where(sources == "ngs")[0].tolist()
    struct_indices = np.where(sources == "struct")[0].tolist()
    if not ngs_indices:
        raise ValueError("Mixed VHH training requires NGS samples")
    if not struct_indices:
        raise ValueError("Mixed VHH training requires struct samples")
    return ngs_indices, struct_indices


class StructRatioIndexSampler(Sampler):
    """Legacy: shuffle-stream indices at struct_sample_ratio (NO length Sortish).

    Kept for reference/tests only. Mixed training must use
    ``MixedSortishStructRatioSampler`` so NGS stays identical to pure-seq.
    """

    def __init__(
        self,
        ngs_indices,
        struct_indices,
        struct_sample_ratio: float,
        group_size: int = 1000,
        num_replicas: int = 1,
        rank: int = 0,
        epoch: int = 0,
    ):
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.ngs_indices = np.asarray(ngs_indices, dtype=np.int64)
        self.struct_indices = np.asarray(struct_indices, dtype=np.int64)
        self.struct_sample_ratio = float(struct_sample_ratio)
        self.group_size = int(group_size)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = epoch
        self.num_samples = int(math.ceil(len(self.ngs_indices) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def _group_counts(self):
        if self.struct_sample_ratio <= 0:
            n_struct = 0
        else:
            n_struct = max(1, int(round(self.group_size * self.struct_sample_ratio)))
        n_struct = min(n_struct, self.group_size)
        return {"struct": n_struct, "ngs": self.group_size - n_struct}

    def __iter__(self):
        """Emit ``total_size`` indices at the configured struct/NGS ratio.

        Cycles both pools so the ratio holds for the whole stream (struct pool
        is tiny). Does NOT length-bucket — do not use for mixed training.
        """
        np.random.seed(self.epoch)
        ngs = self.ngs_indices.copy()
        struct = self.struct_indices.copy()
        np.random.shuffle(ngs)
        np.random.shuffle(struct)
        counts = self._group_counts()
        ngs_i = 0
        struct_i = 0
        indices = []

        def _take_cycled(pool, cursor, k):
            """Take k ids from pool, reshuffling when exhausted."""
            out = []
            while len(out) < k:
                if cursor >= len(pool):
                    np.random.shuffle(pool)
                    cursor = 0
                n = min(k - len(out), len(pool) - cursor)
                out.extend(pool[cursor : cursor + n].tolist())
                cursor += n
            return out, cursor

        while len(indices) < self.total_size:
            remaining = self.total_size - len(indices)
            take_struct = min(counts["struct"], remaining)
            if take_struct:
                chunk, struct_i = _take_cycled(struct, struct_i, take_struct)
                indices.extend(chunk)
                remaining = self.total_size - len(indices)
            take_ngs = min(counts["ngs"], remaining)
            if take_ngs:
                chunk, ngs_i = _take_cycled(ngs, ngs_i, take_ngs)
                indices.extend(chunk)
            if take_struct == 0 and take_ngs == 0:
                break

        if not indices:
            raise RuntimeError("StructRatioIndexSampler produced empty index list")
        indices = indices[: self.total_size]
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class MixedSortishStructRatioSampler(Sampler):
    """Mixed sampler that keeps seq/struct recipes identical to pure training.

    - NGS side: length-bucketed Sortish, same as dummy / sr=0
      (``MappedSortishSampler``).
    - Struct side: for sr>=1, Sortish over struct only (= pure-struct).
      For 0<sr<1, struct is taken in Sortish order (cycled) and appended as a
      *contiguous* block at the end of each NGS Sortish bucket — so ApproxBatch
      packs NGS with NGS and struct with struct (only the bucket boundary may
      mix). Random in-bucket replacement is avoided because it scattered struct
      into NGS length buckets and broke pure-seq packing.
    - Ratio: ~``struct_sample_ratio`` of each bucket is struct.
    """

    def __init__(
        self,
        ngs_indices,
        struct_indices,
        sequence_lengths,
        struct_sample_ratio: float,
        bucket_size: int = 1000,
        num_replicas: int = 1,
        rank: int = 0,
        epoch: int = 0,
    ):
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        self.ngs_indices = np.asarray(ngs_indices, dtype=np.int64)
        self.struct_indices = np.asarray(struct_indices, dtype=np.int64)
        self.sequence_lengths = sequence_lengths
        self.struct_sample_ratio = float(struct_sample_ratio)
        self.bucket_size = int(bucket_size)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = int(epoch)
        self._ngs_sortish = MappedSortishSampler(
            self.ngs_indices,
            sequence_lengths,
            bucket_size=bucket_size,
            num_replicas=num_replicas,
            rank=rank,
            epoch=epoch,
        )
        self._struct_sortish = MappedSortishSampler(
            self.struct_indices,
            sequence_lengths,
            bucket_size=bucket_size,
            num_replicas=num_replicas,
            rank=rank,
            epoch=epoch,
        )

    def _n_struct_per_group(self, group_len: int) -> int:
        sr = self.struct_sample_ratio
        if sr <= 0:
            return 0
        if sr >= 1:
            return group_len
        n = int(round(self.bucket_size * sr))
        n = max(1, n)
        return min(n, group_len)

    def __iter__(self):
        sr = self.struct_sample_ratio
        if sr <= 0.0:
            return iter(self._ngs_sortish)
        if sr >= 1.0:
            return iter(self._struct_sortish)

        self._ngs_sortish.set_epoch(self.epoch)
        self._struct_sortish.set_epoch(self.epoch)

        ngs_stream = [int(i) for i in self._ngs_sortish]
        struct_stream = [int(i) for i in self._struct_sortish]
        if not struct_stream:
            raise RuntimeError("MixedSortishStructRatioSampler: empty struct pool")
        struct_i = 0
        out = []

        for start in range(0, len(ngs_stream), self.bucket_size):
            chunk = ngs_stream[start : start + self.bucket_size]
            k = self._n_struct_per_group(len(chunk))
            if k <= 0:
                out.extend(chunk)
                continue
            # Contiguous split: NGS Sortish prefix + Sortish struct suffix.
            # Keeps NGS packing identical to pure-seq within the prefix.
            ngs_part = chunk[: len(chunk) - k]
            struct_part = []
            for _ in range(k):
                struct_part.append(struct_stream[struct_i % len(struct_stream)])
                struct_i += 1
            # After exhausting one Sortish pass, reshuffle struct order via epoch bump
            # is unnecessary within-epoch; cycling the same Sortish list is fine.
            if struct_i >= len(struct_stream) and struct_i % len(struct_stream) == 0:
                # rotate for variety on subsequent cycles
                struct_stream = struct_stream[1:] + struct_stream[:1]
            out.extend(ngs_part)
            out.extend(struct_part)
        return iter(out)

    def __len__(self):
        if self.struct_sample_ratio >= 1.0:
            return len(self._struct_sortish)
        return len(self._ngs_sortish)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._ngs_sortish.set_epoch(epoch)
        self._struct_sortish.set_epoch(epoch)


def make_dplm2_collater(tokenizer, collater: str = "default"):
    if collater in ("mixed_vhh", "dplm2_mixed_vhh"):
        return MixedVHHCollater(tokenizer)
    if collater in ("dummy_struct", "dplm2_dummy_struct"):
        return DPLM2DummyStructCollater(tokenizer)
    return DPLM2Collater(tokenizer)


def setup_dataloader(
    ds: TokenizedProteinDataset,
    max_tokens=6000,
    bucket_size=1000,
    max_batch_size=100,
    num_workers=8,
    rank=0,
    world_size=1,
    max_len=512,
    tokenizer=None,
    epoch=0,
    collater: str = "default",
    struct_sample_ratio: Optional[float] = None,
) -> DataLoader:
    collater_fn = make_dplm2_collater(tokenizer, collater)
    lens = ds.get_metadata_lens()
    if collater in ("mixed_vhh", "dplm2_mixed_vhh") and struct_sample_ratio is not None:
        ngs_indices, struct_indices = get_mixed_vhh_index_groups(ds)
        # Always Sortish on the active modality stream:
        #   sr<=0 → NGS Sortish (= dummy pure-seq)
        #   sr>=1 → struct Sortish (= pure-struct)
        #   else  → NGS Sortish + replace slots with struct at ratio
        # NEVER use shuffle-only StructRatioIndexSampler here — that broke
        # length bucketing and collapsed IgLM even at 1% struct.
        base_sampler = MixedSortishStructRatioSampler(
            ngs_indices,
            struct_indices,
            lens,
            struct_sample_ratio,
            bucket_size=bucket_size,
            num_replicas=world_size,
            rank=rank,
            epoch=epoch,
        )
    else:
        base_sampler = SortishSampler(
            lens, bucket_size, num_replicas=world_size, rank=rank, epoch=epoch
        )
    train_sampler = ApproxBatchSampler(
        base_sampler,
        max_tokens,
        max_batch_size,
        lens,
        max_len=max_len,
    )
    dl = DataLoader(
        dataset=ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collater_fn,
    )
    return dl


def load_dataset_from_hf(data_path, split):
    split_dir = os.path.join(data_path, split)
    if os.path.isdir(split_dir):
        from datasets import load_from_disk

        return load_from_disk(split_dir)
    ds = load_dataset(data_path, name=split)["train"]
    return ds
