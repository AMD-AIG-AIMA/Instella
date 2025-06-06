# Adapted from ProLong codebase (https://github.com/princeton-nlp/ProLong/tree/main)

import os
import torch

from streaming import StreamingDataset, Stream
import logging

from itertools import islice

from typing import Dict, Any, List, Tuple
from collections.abc import Iterator

from dataclasses import dataclass, field
from typing import Optional, List

import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class StreamingDataArguments:
    single_seq: bool = field(default=False, metadata={"help": "Ignore the document boundaries and treat the whole packed sequence as a single sequence"})
    per_device_max_tokens: Optional[int] = field(default=4_294_967_296, metadata={"help": "Maximum number of tokens per device; this is to avoid some catastrophic cases where the indices or data sequences are not filtered/truncated properly in preprocessing"})
    apply_instruct_masks: bool = field(default=False, metadata={"help": "Whether to apply loss masks over the instructions (for instruction tuning). If enabled, will read the `mask` field in the data and set the corresponding labels to -100."})


class SafeStream(Stream):
    """Safe if multiple processes try to decompress the same shard."""

    def _decompress_shard_part(self, zip_info, zip_filename, raw_filename, compression):
        unique_extension = "." + str(os.getenv("SLURM_JOB_ID", "local")) + "-" + str(os.getpid())
        super()._decompress_shard_part(zip_info, zip_filename, raw_filename + unique_extension, compression)
        os.rename(raw_filename + unique_extension, raw_filename)


class StreamingDataCollator:
    def __init__(self, per_device_max_tokens=262144, single_seq=False, apply_instruct_masks=False):
        self.per_device_max_tokens = per_device_max_tokens
        self.single_seq = single_seq
        self.apply_instruct_masks = apply_instruct_masks

    @torch.no_grad()
    def __call__(self, features):
        all_seq_lengths = []

        all_input_ids = []
        all_doc_lens = []
        all_max_doc_lens = []
        all_label_mask = []

        for item in features:
            input_ids = []
            labels = []
            seq_lengths = []
            label_masks_list = []

            available_tokens = self.per_device_max_tokens

            apply_instruct_masks = self.apply_instruct_masks and ("mask" in item)
            indices = item["indices"] if "indices" in item else [(0, len(item["input_ids"]))]
            if self.single_seq:
                indices = [(0, len(item["input_ids"]))]

            label_seq = torch.tensor(item["input_ids"], dtype=torch.long)

            if "label_mask" in item:
                label_mask = torch.tensor(item["label_mask"], dtype=torch.bool)
            else:
                label_mask = None

            for a, b in indices:
                b = a + min(b - a, available_tokens)
                if b - a > 1:
                    input_seq = torch.tensor(item["input_ids"][a:b], dtype=torch.long)
                    input_ids.append(input_seq)

                    if label_mask is not None:
                        label_masks_list.append(label_mask[a:b])

                    _label = label_seq[a:b]
                    _label[0] = -100
                    if apply_instruct_masks:
                        mask = torch.tensor(item["mask"][a:b], dtype=torch.long)
                        _label[mask == 0] = -100
                    labels.append(_label)

                    seq_lengths.append(b - a)
                    available_tokens -= b - a
                elif available_tokens <= 0:
                    assert available_tokens == 0, "Available tokens should be non-negative"
                    break

            input_ids = torch.concat(input_ids, dim=0)
            labels = torch.concat(labels, dim=0)
            seq_lengths = torch.tensor(seq_lengths, dtype=torch.int32)
            all_seq_lengths.append(seq_lengths)
            if label_mask is not None:
                label_masks_list = torch.concat(label_masks_list, dim=0)
                all_label_mask.append(label_masks_list)

            all_input_ids.append(input_ids) 
        
        max_docs = max((len(seq_len) for seq_len in all_seq_lengths))
        for seq_len in all_seq_lengths:
            doc_pad_shape = (0, max_docs - len(seq_len))
            all_doc_lens.append(F.pad(seq_len, doc_pad_shape, value=0))
            all_max_doc_lens.append(int(seq_len.max()))

        out = {"input_ids": torch.stack(all_input_ids),
               "doc_lens": torch.stack(all_doc_lens),
               "max_doc_lens": all_max_doc_lens}
        
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)

        return out



class SortByLengthDataset(StreamingDataset):
    def __init__(self, *args, sort_by_length_size=1, data_args=None, epoch=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sort_by_length_size = sort_by_length_size
        self.data_args = data_args
        self.total_size = kwargs["epoch_size"] * kwargs["replication"]
        self.epoch = epoch

    def _negative_item_cost(self, item):
        if "indices" in item:
            return -sum(
                (end - start)**2 for start, end in item["indices"]
            )
        elif "length" in item:
            return -item["length"]**2
        else:
            return -len(item["input_ids"])**2

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.sort_by_length_size <= 1:
            yield from super().__iter__()
        else:
            iterator = super().__iter__()
            while True:
                block = list(islice(iterator, self.sort_by_length_size))
                if not block:
                    return

                yield from sorted(block, key=self._negative_item_cost)
