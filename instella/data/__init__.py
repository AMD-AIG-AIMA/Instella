from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from torch.utils.data import DataLoader, DistributedSampler

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig
from ..exceptions import InstellaConfigurationError
from ..torch_util import barrier, get_global_rank, get_world_size
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset
from .streaming_dataset import SortByLengthDataset, SafeStream, StreamingDataCollator

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader"]


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise InstellaConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise InstellaConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_sequence_length,
        memmap_dtype=data_config.effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        eos_token_id=train_config.model.eos_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        generate_doc_lengths=data_config.generate_doc_lengths,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
        instance_filter_config=data_config.instance_filter,
    )

def build_streaming_dataset(paths, train_config: TrainConfig):

    streams = []
    for path in paths:
        if "@" in path:
            path, proportion = path.split("@", 1)
            streams.append(SafeStream(remote=None, local=path, proportion=float(proportion)))
        elif "#" in path:
            path, proportion = path.split("#", 1)
            streams.append(SafeStream(remote=None, local=path, repeat=float(proportion)))
        else:
            streams.append(SafeStream(remote=None, local=path))

    assert isinstance(train_config.max_duration, int)
    epoch_size = int(train_config.global_train_batch_size * train_config.max_duration / train_config.seq_parallel_size)

    num_dataloaders = max(train_config.data.num_workers, 1)
    per_device_step_size = train_config.device_train_batch_size
    per_worker_step_size = per_device_step_size // num_dataloaders

    assert num_dataloaders == 1
    assert per_device_step_size % num_dataloaders == 0, "dataloader workers should divide local batch size"

    return SortByLengthDataset(
        epoch=train_config.epoch or 0,
        streams=streams,
        shuffle=True,
        shuffle_seed=train_config.seed,
        batch_size=train_config.device_train_batch_size,
        epoch_size=epoch_size,
        sort_by_length_size=per_worker_step_size,
        replication=train_config.seq_parallel_size,
    )

def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(
    train_config: TrainConfig,
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    if train_config.data.dataset_type == "MemMapDataset":
        collator = DataCollator(
            pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
        )
        dataset = build_memmap_dataset(
            train_config, train_config.data, include_instance_metadata=include_instance_metadata
        )
        work_dir = Path(train_config.save_folder) / "train_data"
        if get_global_rank() == 0:
            if work_dir.is_dir() and not train_config.save_overwrite:
                raise InstellaConfigurationError(
                    "train data working directory already exists, use --save_overwrite to overwrite"
                )
            else:
                work_dir.mkdir(exist_ok=True, parents=True)
        barrier()
        seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
        return DataLoader(
            IterableDataset(
                dataset,
                train_config.global_train_batch_size,
                seed=seed,
                epoch=train_config.epoch or 0,
                shuffle=True,
                drop_last=train_config.data.drop_last,
                world_size=world_size,
                rank=rank,
                fs_local_rank=fs_local_rank,
                work_dir=work_dir,
            ),
            batch_size=train_config.device_train_batch_size,
            drop_last=train_config.data.drop_last,
            collate_fn=collator,
            num_workers=train_config.data.num_workers,
            pin_memory=train_config.data.pin_memory,
            prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
            persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
            timeout=train_config.data.timeout,
        )
    elif train_config.data.dataset_type == "StreamingDataset":
        dataset = build_streaming_dataset(train_config.data.paths, train_config)
        data_collator = StreamingDataCollator(per_device_max_tokens=train_config.model.max_sequence_length)

        barrier()
        return DataLoader(
            dataset,
            batch_size=train_config.device_train_batch_size,
            collate_fn=data_collator,
            num_workers=1,
            pin_memory=train_config.data.pin_memory,
            persistent_workers=False
        )

