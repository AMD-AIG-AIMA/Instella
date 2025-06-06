import torch
import argparse
import os
from transformers import AutoModelForCausalLM


def write_model(state_dict, output_dir):

    new_state_dict = {}
    
    for k in state_dict:
        if k == "model.embed_tokens.weight":
            new_state_dict["transformer.wte.weight"] = state_dict[k]
        elif "self_attn.k_norm.weight" in k:
            new_state_dict[k.replace("model.layers", "transformer.blocks").replace("self_attn.k_norm.weight", "k_norm.weight")] = state_dict[k]
        elif "self_attn.q_norm.weight" in k:
            new_state_dict[k.replace("model.layers", "transformer.blocks").replace("self_attn.q_norm.weight", "q_norm.weight")] = state_dict[k]
        elif "self_attn.q_proj.weight" in k:
            query = state_dict[k]
            key = state_dict[k.replace("self_attn.q_proj.weight", "self_attn.k_proj.weight")]
            value = state_dict[k.replace("self_attn.q_proj.weight", "self_attn.v_proj.weight")]
            new_k = k.replace("model.layers", "transformer.blocks").replace("self_attn.q_proj.weight", "att_proj.weight")
            new_state_dict[new_k] = torch.cat([query, key, value], dim=0)
        elif "self_attn.k_proj.weight" in k or "self_attn.v_proj.weight" in k:
            continue
        elif "mlp.up_proj.weight" in k:
            up_proj = state_dict[k]
            gate_proj = state_dict[k.replace("mlp.up_proj.weight", "mlp.gate_proj.weight")]
            new_k = k.replace("model.layers", "transformer.blocks").replace("mlp.up_proj.weight", "ff_proj.weight")
            new_state_dict[new_k] = torch.cat([up_proj, gate_proj], dim=0)
        elif "mlp.gate_proj.weight" in k:
            continue
        elif "self_attn.o_proj.weight" in k:
            new_state_dict[k.replace("model.layers", "transformer.blocks").replace("self_attn.o_proj.weight", "attn_out.weight")] = state_dict[k]
        elif "mlp.down_proj.weight" in k:
            new_state_dict[k.replace("model.layers", "transformer.blocks").replace("mlp.down_proj.weight", "ff_out.weight")] = state_dict[k]
        elif "pre_attention_layernorm.weight" in k:
            new_state_dict[k.replace("model.layers", "transformer.blocks").replace("pre_attention_layernorm.weight", "attn_norm.weight")] = state_dict[k]
        elif "pre_feedforward_layernorm.weight" in k:
            new_state_dict[k.replace("model.layers", "transformer.blocks").replace("pre_feedforward_layernorm.weight", "ff_norm.weight")] = state_dict[k]
        elif k == "lm_head.weight":
            new_state_dict["transformer.ff_out.weight"] = state_dict[k]
        elif k == "model.norm.weight":
            new_state_dict["transformer.ln_f.weight"] = state_dict[k]
        
    torch.save(new_state_dict, os.path.join(output_dir, "model.pt"))


def main():
    parser = argparse.ArgumentParser(
            description="Convert HF model weights to unsharded model weights"
        )
    
    parser.add_argument(
        "--hf-model",
        help="HF Model",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        help="Location of output dir.",
        required=True,
    )

    args = parser.parse_args()

    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    state_dict = hf_model.state_dict()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Saving to {args.output_dir}")
    write_model(state_dict, args.output_dir)

if __name__ == '__main__':
    main()