# scripts to merge the checkpoints from second stage pretraining 

import torch
from collections import OrderedDict
import os

def merge_ckpt_weights(ckpt_paths, save_path, merge_strategy="average"):
    """
    Merge the weights of multiple checkpoints into a single checkpoint.

    Args:
        ckpt_paths (list of str): List of paths to the checkpoint files.
        save_path (str): Path to save the merged checkpoint.
        merge_strategy (str): Strategy for merging weights. Options are:
                              - "average": Compute the element-wise average of weights.

    Raises:
        ValueError: If an unsupported merge strategy is provided or if checkpoint files are inconsistent.

    Returns:
        None: The merged checkpoint is saved to `save_path`.
    """
    if not ckpt_paths:
        raise ValueError("No checkpoint paths provided.")

    # Load all state_dicts
    state_dicts = [torch.load(path, map_location="cpu") for path in ckpt_paths]

    # Verify all checkpoints have the same structure
    keys = state_dicts[0].keys()
    for sd in state_dicts:
        if sd.keys() != keys:
            raise ValueError("Checkpoints have inconsistent structures.")

    # Initialize the merged state_dict
    merged_state_dict = OrderedDict()

    if merge_strategy == "average":
        for key in keys:
            # Compute the element-wise average of weights
            merged_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    else:
        raise ValueError(f"Unsupported merge strategy: {merge_strategy}")

    # Save the merged checkpoint
    torch.save(merged_state_dict, save_path)
    print(f"Merged checkpoint saved to {save_path}")


seeds = [5796, 6198, 8915]
ckpts = [f"outputs/pretrain/instella-3b-pretrain-stage2-seed-{seed}/step13727-unsharded/model.pt" for seed in seeds]

save_folder = "outputs/pretrain/instella-3b-pretrain/step13727-unsharded"
os.makedirs(save_folder, exist_ok=True)
merged_ckpt = os.path.join(save_folder, "model.pt")

merge_ckpt_weights(ckpts, merged_ckpt, merge_strategy="average")
