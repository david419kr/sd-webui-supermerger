import gc
import hashlib
import json
import math
import os
import re
import csv
import sys
import traceback
from io import BytesIO
import gradio as gr
import launch
import modules.shared as shared
import numpy as np
import safetensors.torch
import scripts.mergers.components as components
import torch
from modules import extra_networks, scripts, sd_models, launch_utils
from modules.ui import create_refresh_button
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scripts.kohyas import extract_lora_from_models as ext
from scripts.mergers.model_util import filenamecutter, savemodel
from scripts.mergers.mergers import extract_super, unload_forge, q_dequantize, q_quantize, qdtyper, prefixer, BLOCKIDFLUX
from tqdm import tqdm

_forge_tag = launch_utils.git_tag()
forge = _forge_tag[0:2] == "f2" or _forge_tag == "neo"
if forge:
    nets = None
else:
    from scripts.A1111 import networks as nets

selectable = []
pchanged = False

CUDA = torch.device("cuda:0")
CPU = torch.device("cpu")

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
ANIMA_BLOCKID = ["BASE", "IN"] + [f"B{x:02}" for x in range(28)] + [f"L{x:02}" for x in range(6)] + ["LLM", "OUT"]
ANIMA_BLOCKNUM = len(ANIMA_BLOCKID)
BLOCKNUMS = [12,17,20,26,61,ANIMA_BLOCKNUM]
BLOCKIDS=[BLOCKID12,BLOCKID17,BLOCKID20,BLOCKID26]

def to26(ratios):
    if len(ratios) == 26 or len(ratios) == ANIMA_BLOCKNUM or len(ratios) > 40 : return ratios
    ids = BLOCKIDS[BLOCKNUMS.index(len(ratios))]
    output = [0]*26
    for i, id in enumerate(ids):
        output[BLOCKID26.index(id)] = ratios[i]
    return output

def to61s(ratioss):
    out = []
    for ratios in ratioss:
        if len(ratios) == 61:
            out.append(ratios)
        else:
            out.append(ratios + [ratios[0]] * (61 - len(ratios)))
    return out

ANIMA_CHECKPOINT_PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.")
ANIMA_LORA_PREFIXES = ("diffusion_model.", "text_encoders.qwen3_06b.")
ANIMA_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha", ".diff")

def anima_normalize_checkpoint_key(key):
    for prefix in ANIMA_CHECKPOINT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):], prefix
    return key, ""

def anima_is_canonical_checkpoint_keys(keys):
    keyset = set()
    for key in keys:
        normalized, _ = anima_normalize_checkpoint_key(key)
        keyset.add(normalized)
    return (
        "blocks.0.mlp.layer1.weight" in keyset
        and "llm_adapter.blocks.0.cross_attn.q_proj.weight" in keyset
        and "x_embedder.proj.1.weight" in keyset
    )

def anima_is_checkpoint_state_dict(sd):
    return anima_is_canonical_checkpoint_keys(sd.keys())

def anima_lora_module_name(key):
    for suffix in ANIMA_LORA_SUFFIXES:
        if key.endswith(suffix):
            return anima_normalize_lora_module(key[:-len(suffix)])
    return None

def anima_dotted_from_flat_body(body):
    replacements = [
        (r"^llm_adapter_blocks_(\d+)_", r"llm_adapter.blocks.\1."),
        (r"^blocks_(\d+)_", r"blocks.\1."),
        (r"^llm_adapter_(embed|norm|out_proj)", r"llm_adapter.\1"),
        (r"^x_embedder_", r"x_embedder."),
        (r"^t_embedder_", r"t_embedder."),
        (r"^final_layer_", r"final_layer."),
    ]
    for pattern, replacement in replacements:
        body = re.sub(pattern, replacement, body)

    body = body.replace("adaln_modulation_cross_attn_", "adaln_modulation_cross_attn.")
    body = body.replace("adaln_modulation_self_attn_", "adaln_modulation_self_attn.")
    body = body.replace("adaln_modulation_mlp_", "adaln_modulation_mlp.")
    body = body.replace("cross_attn_", "cross_attn.")
    body = body.replace("self_attn_", "self_attn.")
    body = body.replace("mlp_layer1", "mlp.layer1")
    body = body.replace("mlp_layer2", "mlp.layer2")
    body = re.sub(r"(?<=\.)mlp_(\d+)", r"mlp.\1", body)
    body = re.sub(r"t_embedder\.(\d+)_linear_(\d+)", r"t_embedder.\1.linear_\2", body)
    body = body.replace("x_embedder.proj_", "x_embedder.proj.")
    body = body.replace("final_layer.adaln_modulation_", "final_layer.adaln_modulation.")

    for name in ("output_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
        body = body.replace("_" + name, "." + name)
    return body

def anima_normalize_lora_module(module):
    if module is None:
        return None
    if module.startswith("diffusion_model."):
        return module
    if module.startswith("text_encoders.qwen3_06b."):
        return module
    if module.startswith("diffusion_model_"):
        return "diffusion_model." + anima_dotted_from_flat_body(module[len("diffusion_model_"):])
    if module.startswith("lora_unet_"):
        return "diffusion_model." + anima_dotted_from_flat_body(module[len("lora_unet_"):])
    if module.startswith("text_encoders_qwen3_06b_"):
        return "text_encoders.qwen3_06b." + anima_dotted_from_flat_body(module[len("text_encoders_qwen3_06b_"):])
    return module

def anima_normalize_lora_key(key):
    for suffix in ANIMA_LORA_SUFFIXES:
        if key.endswith(suffix):
            module = anima_normalize_lora_module(key[:-len(suffix)])
            return module + suffix
    return key

def normalize_anima_lora_state_dict(sd):
    normalized = {}
    for key, value in sd.items():
        normalized_key = anima_normalize_lora_key(key)
        if normalized_key not in normalized or normalized_key == key:
            normalized[normalized_key] = value
    return normalized

def anima_is_lora_module(module):
    if module is None:
        return False
    for prefix in ANIMA_LORA_PREFIXES:
        if module.startswith(prefix):
            body = module[len(prefix):]
            return body.startswith(("blocks.", "llm_adapter.", "x_embedder.", "t_embedder.", "t_embedding_norm", "final_layer."))
    return False

def anima_is_lora_state_dict(sd):
    for key in sd.keys():
        module = anima_lora_module_name(key)
        if anima_is_lora_module(module):
            return True
    return False

def anima_lora_family_from_state_dict(sd):
    return "Anima" if anima_is_lora_state_dict(sd) else None

def anima_checkpoint_key_from_lora_module(module):
    if module.startswith("diffusion_model."):
        body = module[len("diffusion_model."):]
    elif module.startswith("text_encoders.qwen3_06b."):
        body = module[len("text_encoders.qwen3_06b."):]
    else:
        body = module
    if not body.endswith(".weight"):
        body += ".weight"
    return body

def anima_lora_module_from_checkpoint_key(key):
    body, _ = anima_normalize_checkpoint_key(key)
    if not body.endswith(".weight"):
        return None
    return "diffusion_model." + body[:-len(".weight")]

def anima_checkpoint_key_map(sd):
    out = {}
    for key in sd.keys():
        normalized, _ = anima_normalize_checkpoint_key(key)
        out[normalized] = key
    return out

def anima_block_id_from_checkpoint_key(key):
    body, _ = anima_normalize_checkpoint_key(key)
    if body.startswith(("x_embedder.", "t_embedder.", "t_embedding_norm")):
        return "IN"
    m = re.match(r"blocks\.(\d+)\.", body)
    if m:
        return f"B{int(m.group(1)):02}"
    m = re.match(r"llm_adapter\.blocks\.(\d+)\.", body)
    if m:
        return f"L{int(m.group(1)):02}"
    if body.startswith(("llm_adapter.embed", "llm_adapter.norm", "llm_adapter.out_proj")):
        return "LLM"
    if body.startswith("final_layer."):
        return "OUT"
    return "BASE"

def anima_block_index_from_lora_module(module):
    target_key = anima_checkpoint_key_from_lora_module(module)
    block = anima_block_id_from_checkpoint_key(target_key)
    return ANIMA_BLOCKID.index(block) if block in ANIMA_BLOCKID else 0

def split_lora_name_spec(loranames):
    temp = []
    for n in loranames.split(","):
        if ":" in n:
            temp.append(n.split(":"))
        elif temp:
            temp[-1].append(n)
    return temp

def parse_lora_ratio_presets(loraratios):
    ldict = {}
    for l in loraratios.splitlines():
        if ":" not in l:
            continue
        count = l.count(",")
        if not any(count == x - 1 for x in BLOCKNUMS):
            continue
        ldict[l.split(":", 1)[0].strip()] = l.split(":", 1)[1]
    return ldict

def parse_lora_ratio_spec(parts, preset_dict, family):
    try:
        base = float(parts[1])
    except Exception:
        raise ValueError(f"Invalid LoRA ratio: {':'.join(parts)}")

    if family == "Anima":
        if len(parts) == 2:
            return [base] * ANIMA_BLOCKNUM
        if len(parts) == 3 and parts[2].strip() in preset_dict:
            values = [float(r) * base for r in preset_dict[parts[2].strip()].split(",")]
        elif len(parts[2:]) == ANIMA_BLOCKNUM:
            values = [float(x) for x in parts[2:]]
        else:
            raise ValueError(f"Anima LoRA block weights must be {ANIMA_BLOCKNUM} values: {':'.join(parts)}")
        if len(values) != ANIMA_BLOCKNUM:
            raise ValueError(f"Anima LoRA block weights must be {ANIMA_BLOCKNUM} values, got {len(values)}")
        return values

    if len(parts) == 2:
        return [base] * 26
    if len(parts) == 3:
        if parts[2].strip() in preset_dict:
            ratio = [float(r) * base for r in preset_dict[parts[2].strip()].split(",")]
            return to26(ratio)
        return [base] * 26
    if len(parts[2:]) in BLOCKNUMS:
        ratio = [float(x) for x in parts[2:]]
        return to26(ratio)
    print("ERROR:Number of Blocks must be 12,17,20,26")
    return [base] * 26

def strip_lora_choice_label(name):
    if isinstance(name, str) and "[" in name:
        return name[:name.find("[")]
    return name

def normalize_lora_selection(names):
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    out = []
    for name in names:
        if name not in out:
            out.append(name)
    return out

def lora_choice_for_name(name):
    if name in selectable:
        return name
    base_name = strip_lora_choice_label(name)
    for choice in selectable:
        if strip_lora_choice_label(choice) == base_name:
            return choice
    return None

def valid_lora_selection(names):
    out = []
    for name in normalize_lora_selection(names):
        choice = lora_choice_for_name(name)
        if choice is not None and choice not in out:
            out.append(choice)
    return out

def lora_choices_selected_first(names):
    selected = valid_lora_selection(names)
    selected_set = set(selected)
    return selected + [name for name in selectable if name not in selected_set]

def lora_selection_to_text(names, ratio):
    clean_names = [strip_lora_choice_label(name) for name in normalize_lora_selection(names)]
    if not clean_names:
        return ""
    return f":{ratio},".join(clean_names)+f":{ratio} "

def lora_dropdown_update(names):
    selected = valid_lora_selection(names)
    return gr.update(choices=lora_choices_selected_first(selected), value=selected)

def lora_dropdown_update_and_text(names, ratio):
    selected = valid_lora_selection(names)
    return gr.update(choices=lora_choices_selected_first(selected), value=selected), lora_selection_to_text(selected, ratio)

def load_lora_header_or_state(filename):
    return load_state_header(filename, torch.float)

def load_anima_checkpoint_state_dict(path, device="cpu"):
    if os.path.splitext(path)[1] == ".safetensors":
        return load_file(path, device=device)
    return torch.load(path, map_location=device)

def checkpoint_file_is_anima(path):
    if os.path.splitext(path)[1] == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as f:
            return anima_is_canonical_checkpoint_keys(f.keys())
    return anima_is_checkpoint_state_dict(load_anima_checkpoint_state_dict(path, device="cpu"))

def tensor_shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, dict):
        return tuple(value.get("shape", []))
    return ()

def tensor_rank(value):
    shape = tensor_shape(value)
    return shape[0] if len(shape) > 0 else None

def anima_lora_modules(sd):
    modules = set()
    for key in sd.keys():
        if key.endswith(".lora_down.weight") or key.endswith(".diff"):
            module = anima_lora_module_name(key)
            if anima_is_lora_module(module):
                modules.add(module)
    return modules

def anima_lora_module_rank(sd, module):
    value = sd.get(module + ".lora_down.weight", None)
    return tensor_rank(value)

def anima_lora_module_diff(sd, module, device, calc_dtype):
    diff_key = module + ".diff"
    if diff_key in sd:
        return sd[diff_key].to(device=device, dtype=calc_dtype)

    down_key = module + ".lora_down.weight"
    up_key = module + ".lora_up.weight"
    if down_key not in sd or up_key not in sd:
        return None

    down_weight = sd[down_key].to(device=device, dtype=calc_dtype)
    up_weight = sd[up_key].to(device=device, dtype=calc_dtype)
    dim = down_weight.size(0)
    alpha = sd.get(module + ".alpha", torch.tensor(dim))
    if isinstance(alpha, torch.Tensor):
        alpha = float(alpha.detach().cpu().item())
    scale = alpha / dim if dim else 0.0

    if len(down_weight.shape) == 2:
        return (up_weight @ down_weight) * scale
    if len(down_weight.shape) == 4:
        return torch.nn.functional.conv2d(
            down_weight.permute(1, 0, 2, 3), up_weight
        ).permute(1, 0, 2, 3) * scale
    return None

def anima_svd_to_lora(out_sd, module, mat, rank, device):
    mat = mat.to(device=device, dtype=torch.float32)
    if len(mat.shape) == 1:
        out_sd[module + ".diff"] = mat.to("cpu").contiguous()
        return "diff"
    if len(mat.shape) != 2:
        return None

    out_dim, in_dim = mat.shape
    rank = min(int(rank), int(in_dim), int(out_dim))
    if rank <= 0:
        return None

    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)
    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val
    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    out_sd[module + ".lora_up.weight"] = U.to("cpu").contiguous()
    out_sd[module + ".lora_down.weight"] = Vh.to("cpu").contiguous()
    out_sd[module + ".alpha"] = torch.tensor(float(rank))
    return "lora"

def anima_base_metadata(name, dim, save_precision, extra=None):
    metadata = {
        "ss_base_model_version": "anima",
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(dim),
        "ss_mixed_precision": save_precision,
        "ss_output_name": name,
        "ss_supermerger_anima_lora": "1",
    }
    if extra:
        metadata.update({k: str(v) for k, v in extra.items()})
    return metadata

def make_anima_lora_from_checkpoints(model_a, model_b, dim, saveto, alpha, beta, save_precision, device):
    path_a = fullpathfromname(model_a)
    path_b = fullpathfromname(model_b)
    sd_a = load_anima_checkpoint_state_dict(path_a, device="cpu")
    sd_b = load_anima_checkpoint_state_dict(path_b, device="cpu")

    if not anima_is_checkpoint_state_dict(sd_a) or not anima_is_checkpoint_state_dict(sd_b):
        return None

    map_a = anima_checkpoint_key_map(sd_a)
    map_b = anima_checkpoint_key_map(sd_b)
    common = sorted(k for k in map_a.keys() if k in map_b and k.endswith(".weight") and sd_a[map_a[k]].shape == sd_b[map_b[k]].shape)

    calc_device = CUDA if "cuda" in str(device) and torch.cuda.is_available() else CPU
    out_sd = {}
    lora_count = 0
    diff_count = 0
    rank = 128 if type(dim) != int else int(dim)

    for key in tqdm(common, desc="Anima LoRA SVD"):
        module = anima_lora_module_from_checkpoint_key(key)
        if module is None:
            continue
        a = sd_a[map_a[key]]
        b = sd_b[map_b[key]]
        if len(a.shape) not in (1, 2):
            continue
        mat = (float(alpha) * a.to(device=calc_device, dtype=torch.float32)) - (float(beta) * b.to(device=calc_device, dtype=torch.float32))
        kind = anima_svd_to_lora(out_sd, module, mat, rank, calc_device)
        if kind == "lora":
            lora_count += 1
        elif kind == "diff":
            diff_count += 1
        del mat

    if lora_count + diff_count == 0:
        return "ERROR: No Anima LoRA eligible tensors were found"

    metadata = anima_base_metadata(
        os.path.splitext(os.path.basename(saveto))[0],
        rank,
        save_precision,
        {"sshs_recipe": f"{alpha} * {model_a} - {beta} * {model_b}", "sshs_lora_count": lora_count, "sshs_diff_count": diff_count},
    )
    save_to_file(saveto, out_sd, out_sd, str_to_dtype(save_precision), metadata)
    return f"Anima LoRA weights are saved to: {saveto} ({lora_count} LoRA tensors, {diff_count} diff tensors)"

def merge_anima_lora_models(models, ratios, new_rank, save_precision, calc_precision, device, extract=False, alpha=1, beta=1, smooth=1):
    calc_device = CUDA if "cuda" in str(device) and torch.cuda.is_available() else CPU
    calc_dtype = str_to_dtype(calc_precision) or torch.float32
    if calc_device == CPU:
        calc_dtype = torch.float32
    lora_sds = []
    ranks = []

    for model in models:
        sd, _, _ = load_state_dict(model, calc_dtype, "cpu")
        sd = normalize_anima_lora_state_dict(sd)
        if not anima_is_lora_state_dict(sd):
            raise ValueError(f"Non-Anima LoRA cannot be mixed with Anima LoRA: {model}")
        lora_sds.append(sd)
        ranks += [r for module in anima_lora_modules(sd) for r in [anima_lora_module_rank(sd, module)] if r]

    modules = sorted(set().union(*[anima_lora_modules(sd) for sd in lora_sds]))
    if not modules:
        raise ValueError("No Anima LoRA modules were found")

    if isinstance(new_rank, int) and new_rank > 0:
        rank = new_rank
    elif ranks:
        rank = max(ranks)
    else:
        rank = 128

    out_sd = {}
    for module in tqdm(modules, desc="Anima LoRA merge"):
        merged = None
        if extract:
            if len(lora_sds) < 2:
                raise ValueError("Extract from two LoRAs requires at least two LoRAs")
            mat_a = anima_lora_module_diff(lora_sds[0], module, calc_device, calc_dtype)
            mat_b = anima_lora_module_diff(lora_sds[1], module, calc_device, calc_dtype)
            if mat_a is None and mat_b is None:
                continue
            if mat_a is None:
                mat_a = torch.zeros_like(mat_b)
            if mat_b is None:
                mat_b = torch.zeros_like(mat_a)
            if mat_a.shape != mat_b.shape:
                raise ValueError(f"Anima LoRA shape mismatch: {module} {tuple(mat_a.shape)} != {tuple(mat_b.shape)}")
            block_index = anima_block_index_from_lora_module(module)
            mat_a = mat_a * ratios[0][block_index]
            mat_b = mat_b * ratios[1][block_index]
            merged = extract_super(None, mat_a, mat_b, alpha, beta, smooth)
        else:
            for sd, ratio in zip(lora_sds, ratios):
                mat = anima_lora_module_diff(sd, module, calc_device, calc_dtype)
                if mat is None:
                    continue
                block_ratio = ratio[anima_block_index_from_lora_module(module)]
                if merged is None:
                    merged = mat * block_ratio
                else:
                    if merged.shape != mat.shape:
                        raise ValueError(f"Anima LoRA shape mismatch: {module} {tuple(merged.shape)} != {tuple(mat.shape)}")
                    merged += mat * block_ratio
        if merged is None:
            continue
        anima_svd_to_lora(out_sd, module, merged, rank, calc_device)
        del merged

    if not out_sd:
        raise ValueError("No Anima LoRA tensors were merged")
    return out_sd

def pluslora_anima(theta_0, filenames, ratios, calc_precision, device):
    calc_device = CUDA if "cuda" in str(device) and torch.cuda.is_available() else CPU
    calc_dtype = str_to_dtype(calc_precision) or torch.float32
    if calc_device == CPU:
        calc_dtype = torch.float32
    key_map = anima_checkpoint_key_map(theta_0)
    applied = 0

    for filename, ratio in zip(filenames, ratios):
        lora_sd, _, _ = load_state_dict(filename, calc_dtype, "cpu")
        lora_sd = normalize_anima_lora_state_dict(lora_sd)
        if not anima_is_lora_state_dict(lora_sd):
            raise ValueError(f"Non-Anima LoRA cannot be baked into an Anima checkpoint: {filename}")
        for module in tqdm(sorted(anima_lora_modules(lora_sd)), desc=f"Anima bake {os.path.basename(filename)}"):
            target_key = anima_checkpoint_key_from_lora_module(module)
            if target_key not in key_map:
                continue
            mat = anima_lora_module_diff(lora_sd, module, calc_device, calc_dtype)
            if mat is None:
                continue
            block_ratio = ratio[anima_block_index_from_lora_module(module)]
            if block_ratio == 0:
                continue
            actual_key = key_map[target_key]
            weight = theta_0[actual_key]
            if tuple(weight.shape) != tuple(mat.shape):
                raise ValueError(f"Anima checkpoint shape mismatch: {actual_key} {tuple(weight.shape)} != {tuple(mat.shape)}")
            merged = weight.to(device=calc_device, dtype=calc_dtype) + mat * block_ratio
            theta_0[actual_key] = merged.to(device="cpu", dtype=weight.dtype).contiguous()
            applied += 1
            del mat, merged

    if applied == 0:
        raise ValueError("No Anima LoRA tensors were applied to the checkpoint")
    print(f"SuperMerger: baked {applied} Anima LoRA tensors into checkpoint.")
    return theta_0

def f_changediffusers(version):
    launch.run_pip(f"install diffusers=={version}", f"diffusers ver {version}")

def on_ui_tabs():
    import lora
    global selectable
    selectable= [x[0] for x in lora.available_loras.items()]
    sml_path_root = scripts.basedir()
    LWEIGHTSPRESETS="\
    NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
    ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n\
    INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
    IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
    INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
    MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\n\
    OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\n\
    OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\n\
    OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n\
    ALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
    lbwpath = os.path.join(sml_path_root,"scripts", "lbwpresets.txt")
    lbwpathn = os.path.join(sml_path_root,"extensions","sd-webui-lora-block-weight","scripts", "lbwpresets.txt")
    dimpath = os.path.join(sml_path_root,"extensions","sd-webui-supermerger","loradims.csv")
    sml_lbwpresets=""

    if os.path.isfile(lbwpath):
        with open(lbwpath,encoding="utf-8") as f:
            sml_lbwpresets = f.read()
    elif os.path.isfile(lbwpathn):
        with open(lbwpathn,encoding="utf-8") as f:
            sml_lbwpresets = f.read()
    else:
        sml_lbwpresets=LWEIGHTSPRESETS

    try:
        import diffusers
        d_ver = diffusers.__version__
    except:
        d_ver = None

    with gr.Blocks(analytics_enabled=False) :
        sml_submit_result = gr.Textbox(label="Message")
        with gr.Row(equal_height=False):
            with gr.Column():
                sml_cpmerge = gr.Button(elem_id="model_merger_merge", value="Merge to Checkpoint(Model A)",variant='primary')
                sml_merge = gr.Button(elem_id="model_merger_merge", value="Merge LoRAs",variant='primary')
                with gr.Row(equal_height=False):
                    sml_settings = gr.CheckboxGroup(["same to Strength", "overwrite"], label="settings")
                    sml_filename = gr.Textbox(label="filename(option)",lines=1,visible =True,interactive  = True)  
                sml_metasettings = gr.Radio(value = "create new",choices = ["create new","create new without output_name", "merge","save all", "use first lora"], label="metadata")
                with gr.Row(equal_height=False):
                    save_precision = gr.Radio(label = "save precision",choices=["float","fp16","bf16"],value = "fp16",type="value")
                    calc_precision = gr.Radio(label = "calc precision(fp16:cuda only)" ,choices=["float","fp16","bf16"],value = "float",type="value")
                    device = gr.Radio(label = "device",choices=["cuda","cpu"],value = "cuda",type="value")
            with gr.Column():
                sml_makelora = gr.Button(elem_id="model_merger_merge", value="Make LoRA (alpha * Model A - beta * Model B)",variant='primary')
                sml_extract = gr.Button(elem_id="model_merger_merge", value="Extract from two LoRAs",variant='primary')
                with gr.Row(equal_height=False):
                    sml_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model A",interactive=True)
                    create_refresh_button(sml_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                with gr.Row(equal_height=False):
                    sml_model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model B",interactive=True)
                    create_refresh_button(sml_model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                with gr.Row(equal_height=False):
                    alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=1)
                    beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=1)
                    smooth = gr.Slider(label="gamma(smooth)", minimum=-1, maximum=20, step=0.1, value=1)
        
        sml_dim = gr.Radio(label = "remake dimension",choices = ["no","auto",4,8,16,32,64,128,256,512,768,1024],value = "no",type = "value")
        sml_loranames = gr.Textbox(label='LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,...(":blocks" is option, not necessary)',lines=1,value="",visible =True)
        sml_loras = gr.Dropdown(label = "LoRAs on disk",choices = selectable,value=[],type="value",multiselect=True,filterable=True,interactive=True,visible = True,elem_id="sml_loras_dropdown")
        sml_loratypes = gr.CheckboxGroup(show_label=False, choices= ["LoRA", "LoCon", "Others"], value=["LoRA", "LoCon", "Others"])
        sml_dims = gr.CheckboxGroup(label = "1.X/2.X",choices=[],value = [],type="value",interactive=True,visible = False)
        sml_dims_xl = gr.CheckboxGroup(label = "XL",choices=[],value = [],type="value",interactive=True,visible = False)
        sml_dims_flux = gr.CheckboxGroup(label = "Flux",choices=[],value = [],type="value",interactive=True,visible = False)
        sml_dims_anima = gr.CheckboxGroup(label = "Anima",choices=[],value = [],type="value",interactive=True,visible = False)
        with gr.Row(equal_height=False):
            sml_calcdim = gr.Button(elem_id="calcloras", value="Calculate LoRA dimensions",variant='primary')
            sml_calcsets = gr.CheckboxGroup(choices=["Save as CSV","Load from CSV"],show_label=False)
            sml_update = gr.Button(elem_id="calcloras", value="update list",variant='primary')
            sml_lratio = gr.Slider(label="default LoRA multiplier", minimum=-1.0, maximum=2, step=0.1, value=1)

        with gr.Row():
            sml_selectall = gr.Button(elem_id="sml_selectall", value="select all",variant='primary')
            sml_deselectall = gr.Button(elem_id="slm_deselectall", value="deselect all",variant='primary')
            components.frompromptb = gr.Button(elem_id="slm_deselectall", value="get from prompt",variant='primary')
            hidenb = gr.Checkbox(value = False,visible = False)
        sml_loraratios = gr.TextArea(label="",value=sml_lbwpresets,visible =True,interactive  = True)

        with gr.Row():
            changediffusers = gr.Button(elem_id=f"change_diffusers_version", value=f"change diffusers version(now:{d_ver})",variant='primary')
            dversion = gr.Textbox(label="diffusers version",lines=1,visible =True,interactive  = True)
        components.sml_loranames = [sml_loras, sml_loranames, hidenb]

        changediffusers.click(
            fn=f_changediffusers,
            inputs=[dversion],
            outputs=[sml_submit_result]
        )

        sml_merge.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_dim,save_precision,calc_precision,sml_metasettings,alpha,beta,smooth,gr.Checkbox(value = True,visible = False),device],
            outputs=[sml_submit_result]
        )

        sml_extract.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_dim,save_precision,calc_precision,sml_metasettings,alpha,beta,smooth,gr.Checkbox(value = False,visible = False),device],
            outputs=[sml_submit_result]
        )

        sml_makelora.click(
            fn=makelora,
            inputs=[sml_model_a,sml_model_b,sml_dim,sml_filename,sml_settings,alpha,beta,save_precision,calc_precision,sml_metasettings,device],
            outputs=[sml_submit_result]
        )

        sml_cpmerge.click(
            fn=pluslora,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_model_a,save_precision,calc_precision,sml_metasettings,device],
            outputs=[sml_submit_result]
        )

        ldict = {}

        def toselect(input):
            out = []
            for name, vals in input.items():
                if (not isinstance(vals, list)) or len(vals) != 3: continue
                dim, ltype, sdver = vals
                add = [] if dim == "LyCORIS" else [str(dim)]
                if ltype != "LoRA": add +=[ltype]
                if sdver != "1.X/2.X": add += [sdver]
                out.append(f"{name}[{','.join(add)}]" if add != ["","",""] else f"{name}")
            return out

        def updateloras(current_selection, ratio):
            lora.list_available_loras()
            names = []
            for n in  lora.available_loras.items():
                if n[0] not in ldict:ldict[n[0]] = ["","",""]
                names.append(n[0])

            for l in list(ldict.keys()):
                if l not in names:ldict.pop(l)

            global selectable
            selectable = toselect(ldict)
            return lora_dropdown_update_and_text(current_selection, ratio)

        sml_update.click(fn = updateloras,inputs=[sml_loras,sml_lratio],outputs = [sml_loras,sml_loranames])

        def makedimlist(ver):
            outs = []
            outs_list = []
            for dim, _, sdver in ldict.values():
                if sdver == ver or (ver == "1.X/2.X" and dim == "unknown"):
                    if isinstance(dim, int):
                        if dim not in outs:
                            outs.append(dim)
                    else:
                        if dim not in outs_list:
                            outs_list.append(dim)
            outs = sorted(set(outs))
            return outs + outs_list

        def calculatedim(calcsets, device, current_selection, ratio):
            # CSVから読み込む
            if "Load from CSV" in calcsets:
                with open(dimpath, mode='r', encoding='utf-8') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        ldict[row[0]] = row[1:]

            print("listing dimensions...")
            for n in tqdm(lora.available_loras.items()):
                name = n[0] 
                if name in ldict and ldict[n[0]] != ["","",""]:
                    continue
                c_lora = lora.available_loras.get(n[0], None) 
                
                try:
                    d, t, s = dimgetter(c_lora.filename, device)
                except:
                    d, t, s = dimgetter(c_lora.filename)
                    
                ldict[name] = [d,t,s]

            # CSVに保存
            if "Save as CSV" in calcsets:
                with open(dimpath, mode='w', encoding='utf-8', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for key, value in ldict.items():
                        csv_writer.writerow([key, *value])

            global selectable
            selectable = toselect(ldict)
            dropdown_update, loranames_update = lora_dropdown_update_and_text(current_selection, ratio)
            return (dropdown_update,
                    loranames_update,
                    gr.update(visible=True, choices=makedimlist("1.X/2.X")),
                    gr.update(visible=True, choices=makedimlist("XL")),
                    gr.update(visible=True, choices=makedimlist("Flux")),
                    gr.update(visible=True, choices=makedimlist("Anima"))
            )

        sml_calcdim.click(
            fn=calculatedim,
            inputs=[sml_calcsets, device, sml_loras, sml_lratio],
            outputs=[sml_loras,sml_loranames,sml_dims,sml_dims_xl,sml_dims_flux,sml_dims_anima]
        )

        def dimselector(dims, dims_xl, dims_flux, dims_anima, ltypes, current_selection, ratio):
            rl={}
            ltypes = list(ltypes or [])
            if "Others" in ltypes:ltypes += ["LyCORIS", "unknown"]
            for name, vals in ldict.items():
                dim, ltype, sdver = vals
                if sdver == "1.X/2.X" and dim in dims and ltype in ltypes:
                    rl[name] = vals
                if sdver == "XL" and dim in dims_xl and ltype in ltypes:
                    rl[name] = vals
                if sdver == "Flux" and dim in dims_flux and ltype in ltypes:
                    rl[name] = vals
                if sdver == "Anima" and dim in dims_anima and ltype in ltypes:
                    rl[name] = vals

            global selectable
            selectable = toselect(rl)

            return lora_dropdown_update_and_text(current_selection, ratio)

        def select_all_loras(ratio):
            names = list(selectable)
            return gr.update(choices=names, value=names), lora_selection_to_text(names, ratio)

        def deselect_all_loras():
            return gr.update(choices=selectable, value=[]), ""

        def llister(names,ratio, hiden):
          if hiden:return gr.update()
          return lora_selection_to_text(names, ratio)

        sml_selectall.click(fn = select_all_loras,inputs=[sml_lratio],outputs = [sml_loras,sml_loranames])
        sml_deselectall.click(fn = deselect_all_loras,outputs = [sml_loras,sml_loranames])
        hidenb.change(fn=lambda x: False, outputs = [hidenb])
        sml_loras.change(fn=llister,inputs=[sml_loras,sml_lratio, hidenb],outputs=[sml_loranames])
        sml_loras.blur(fn=lora_dropdown_update,inputs=[sml_loras],outputs=[sml_loras])
        sml_dims.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_dims_anima,sml_loratypes,sml_loras,sml_lratio],outputs=[sml_loras,sml_loranames])
        sml_dims_xl.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_dims_anima,sml_loratypes,sml_loras,sml_lratio],outputs=[sml_loras,sml_loranames])
        sml_dims_flux.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_dims_anima,sml_loratypes,sml_loras,sml_lratio],outputs=[sml_loras,sml_loranames])
        sml_dims_anima.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_dims_anima,sml_loratypes,sml_loras,sml_lratio],outputs=[sml_loras,sml_loranames])
        sml_loratypes.change(fn=dimselector,inputs=[sml_dims,sml_dims_xl,sml_dims_flux,sml_dims_anima,sml_loratypes,sml_loras,sml_lratio],outputs=[sml_loras,sml_loranames])

##############################################################
####### make LoRA from checkpoint

def makelora(model_a,model_b,dim,saveto,settings,alpha,beta,save_precision,calc_precision,metasets,device):
    print("make LoRA start")
    if model_a == "" or model_b =="":
      return "ERROR: No model Selected"
    gc.collect()

    try:
        currentinfo = shared.sd_model.sd_checkpoint_info
    except:
        currentinfo = None

    lowvramdealer() #web-uiのバグ対策

    if saveto =="" : saveto = makeloraname(model_a,model_b)
    if not ".safetensors" in saveto :saveto  += ".safetensors"
    saveto = os.path.join(shared.cmd_opts.lora_dir,saveto)

    dim = 128 if type(dim) != int else int(dim)
    if os.path.isfile(saveto ) and not "overwrite" in settings:
        _err_msg = f"Output file ({saveto}) existed and was not saved"
        print(_err_msg)
        return _err_msg

    model_a_path = fullpathfromname(model_a)
    model_b_path = fullpathfromname(model_b)
    a_is_anima = checkpoint_file_is_anima(model_a_path)
    b_is_anima = checkpoint_file_is_anima(model_b_path)
    if a_is_anima or b_is_anima:
        if not (a_is_anima and b_is_anima):
            return "ERROR: Anima and non-Anima checkpoints cannot be mixed for LoRA extraction"
        return make_anima_lora_from_checkpoints(model_a, model_b, dim, saveto, alpha, beta, save_precision, device)

    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
    load_model(checkpoint_info)

    model = shared.sd_model
    print(type(model).__name__)
    print("XL" in type(model).__name__)

    is_sdxl = type(model).__name__ == "StableDiffusionXL" or getattr(model,'is_sdxl', False)
    is_sd2 = type(model).__name__ == "StableDiffusion2" or getattr(model,'is_sd2', False)
    is_sd1 = type(model).__name__ == "StableDiffusion" or getattr(model,'is_sd1', False)
    is_flux = type(model).__name__ == "Flux" or getattr(model,'is_flux', False)

    print(f"Detected model type: SDXL: {is_sdxl}, SD2.X: {is_sd2}, SD1.X: {is_sd1}")

    if forge:
        unload_forge()
    else:
        sd_models.unload_model_weights()

    args = Kohya_extract_args(
        v2=is_sd2,
        v_parameterization=True,
        sdxl=is_sdxl,
        save_precision=save_precision,
        model_org=fullpathfromname(model_b),
        model_tuned=fullpathfromname(model_a),
        save_to=saveto,
        dim=dim,
        conv_dim=None,
        device=device,
        no_metadata=False,
        alpha = alpha,
        beta = beta
    )

    result = ext.svd(args)

    if currentinfo:
        load_model(currentinfo)
    return result

##############################################################
####### merge LoRAs

def lmerge(loranames,loraratioss,settings,filename,dim,save_precision,calc_precision,metasets,alpha,beta,smooth,merge,device):
    try:
        import lora
        parsed_names = split_lora_name_spec(loranames)
        if not parsed_names:
            return "ERROR: No LoRA Selected"

        alias_map = getattr(lora, "available_lora_aliases", {})
        selected = []
        for n in parsed_names:
            c_lora = lora.available_loras.get(n[0], alias_map.get(n[0], None))
            if c_lora is None:
                lora.list_available_loras()
                c_lora = lora.available_loras.get(n[0], alias_map.get(n[0], None))
            if c_lora is None:
                return f"ERROR: LoRA not found: {n[0]}"
            header = load_lora_header_or_state(c_lora.filename)
            selected.append((n, c_lora, anima_lora_family_from_state_dict(header)))

        if any(family == "Anima" for _, _, family in selected):
            if not all(family == "Anima" for _, _, family in selected):
                return "ERROR: Anima LoRA and non-Anima LoRA cannot be mixed"

            preset_dict = parse_lora_ratio_presets(loraratioss)
            ln, lr, lm = [], [], []
            for n, c_lora, _ in selected:
                ratio = parse_lora_ratio_spec(n, preset_dict, "Anima")
                ln.append(c_lora.filename)
                lr.append(ratio)
                lm.append(prepare_merge_metadata(n[1], ",".join([str(x) for x in ratio]), c_lora))

            if filename =="":filename =loranames.replace(",","+").replace(":","_")
            if not ".safetensors" in filename:filename += ".safetensors"
            loraname = filename.replace(".safetensors", "")
            filename = os.path.join(shared.cmd_opts.lora_dir,filename)
            if os.path.isfile(filename) and not "overwrite" in settings:
                _err_msg = f"Output file ({filename}) existed and was not saved"
                print(_err_msg)
                return _err_msg

            new_rank = int(dim) if dim != "no" and dim != "auto" else 0
            if merge:
                sd = merge_anima_lora_models(ln, lr, new_rank, save_precision, calc_precision, device)
            else:
                if len(ln) < 2:
                    return "ERROR: Extract from two LoRAs requires at least two LoRAs"
                sd = merge_anima_lora_models(ln[:2], lr[:2], new_rank, save_precision, calc_precision, device, extract=True, alpha=alpha, beta=beta, smooth=smooth)

            metadata = create_merge_metadata(sd, lm, loraname, save_precision, metasets)
            metadata["ss_base_model_version"] = "anima"
            metadata["ss_supermerger_anima_lora"] = "1"
            save_to_file(filename, sd, sd, str_to_dtype(save_precision), metadata)
            del sd
            gc.collect()
            torch.cuda.empty_cache()
            return "saved : "+filename

        loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]
        if any([x is None for x in loras_on_disk]):
            lora.list_available_loras()

            loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]

        lnames = loranames.split(",")

        #LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,.
        #LoRAname1:ratio1:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,LoRAname2:ratio2:Blocks2,.

        temp = []
        for n in lnames:
            if ":" in n:
                temp.append(n.split(":"))
            else:
                temp[-1].append(n)

        lnames = temp

        loraratios=loraratioss.splitlines()
        ldict ={}

        for i,l in enumerate(loraratios):
            if ":" not in l or not any(l.count(",") == x - 1 for x in BLOCKNUMS) : continue
            ldict[l.split(":")[0]]=l.split(":")[1]

        ln, lr, ld, lt, lm, ls = [], [], [], [], [], [] #lm: 各LoRAのマージ用メタデータ #ls: SD-?
        dmax = 1

        for i,n in enumerate(lnames):
            if len(n) ==2:
                ratio = [float(n[1])]*26
            elif len(n) ==3:
                if n[2].strip() in ldict:
                    ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
                    ratio = to26(ratio)
                else:ratio = [float(n[1])]*26
            elif len(n[2:]) in BLOCKNUMS:
                ratio = [float(x) for x in n[2:]]
                ratio = to26(ratio)
            else:
                print("ERROR:Number of Blocks must be 12,17,20,26")
                ratio = [float(n[1])]*26
            c_lora = lora.available_loras.get(n[0], None) 
            ln.append(c_lora.filename)
            lr.append(ratio)
            d, t, s = dimgetter(c_lora.filename, device)
            if t == "LoCon" and isinstance(d, list):
                d = list(set(d))
                d = d[0]
            lt.append(t)
            ld.append(d)
            ls.append(s)
            if d != "LyCORIS" and isinstance(d, int):
                if d > dmax : dmax = d
            
            # LoRA毎のメタデータを保存
            meta = prepare_merge_metadata( n[1], ",".join( [str(n) for n in ratio] ), c_lora )
            lm.append( meta )

        if filename =="":filename =loranames.replace(",","+").replace(":","_")
        if not ".safetensors" in filename:filename += ".safetensors"
        loraname = filename.replace(".safetensors", "")
        filename = os.path.join(shared.cmd_opts.lora_dir,filename)

        auto = True if dim == "auto" else False
    
        dim = int(dim) if dim != "no" and dim != "auto" else 0

        if merge:
            if "LyCORIS" in ld:
                if len(ld) !=1:
                    return "multiple merge of LyCORIS is not supported"
                sd = lycomerge(ln[0], lr[0], calc_precision, device)
            elif dim > 0:
                print("change demension to ", dim)
                sd = merge_lora_models_dim(ln, lr, dim,settings,device,calc_precision)
            elif auto and ld.count(ld[0]) != len(ld):
                print("change demension to ",dmax)
                sd = merge_lora_models_dim(ln, lr, dmax,settings,device,calc_precision)
            else:
                sd = merge_lora_models(ln, lr, settings, False, calc_precision, device)

            if os.path.isfile(filename) and not "overwrite" in settings:
                _err_msg = f"Output file ({filename}) existed and was not saved"
                print(_err_msg)
                return _err_msg
        else:
            a = merge_lora_models(ln[0:1], lr[0:1], settings, False, calc_precision, device)
            b = merge_lora_models(ln[1:2], lr[1:2], settings, False, calc_precision, device)
            sd = extract_two(a,b,alpha,beta,smooth)
        
        # マージ後のメタデータを取得
        metadata = create_merge_metadata( sd, lm, loraname, save_precision,metasets )

        save_to_file(filename,sd,sd, str_to_dtype(save_precision), metadata)
        sd = None
        del sd
        gc.collect()
        torch.cuda.empty_cache()

        return "saved : "+filename
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        return exc_value

def merge_lora_models(models, ratios, sets, locon, calc_precision, device):
    base_alphas = {}                          # alpha for merged model
    base_dims = {}
    merge_dtype = str_to_dtype(calc_precision)
    merged_sd = {}
    fugou = 1
    for model, ratios in zip(models, ratios):
        keylist = LBLCOKS26

        print(f"merging {model}: {ratios}")
        lora_sd, metadata, isv2 = load_state_dict(model, merge_dtype, device)

        # get alpha and dim
        alphas = {}                             # alpha for current model
        dims = {}                               # dims for current model

        base_dims, base_alphas, dims, alphas = dimalpha(lora_sd, base_dims, base_alphas)

        print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        print(f"merging...")
        for key in lora_sd.keys():
            if 'alpha' in key or "dora" in key:
                continue

            lora_module_name = key[:key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            ratio = ratios[blockfromkey(key, keylist, isv2)]
            #print(key,blockfromkey(key, keylist, isv2))
            
            if "same to Strength" in sets:
                ratio, fugou = (ratio ** 0.5, 1) if ratio > 0 else (abs(ratio) ** 0.5, -1)

            if "lora_down" in key:
                ratio = ratio * fugou

            scale = math.sqrt(alpha / base_alpha) * ratio

            if key in merged_sd:
                assert merged_sd[key].size() == lora_sd[key].size(), (
                    f"weights shape mismatch merging v1 and v2, different dims? "
                    f"/ 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                    f" {merged_sd[key].size()} ,{lora_sd[key].size()}, {lora_module_name}"
                )
                merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale
        del lora_sd

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    return merged_sd

def merge_lora_models_dim(models, ratios, new_rank, sets, device, calc_precision):
    CHUNK_SIZE = 50

    isv2 = False
    merge_dtype = str_to_dtype(calc_precision)
    
    lora_sds = []
    print("Loading LoRA models...")
    for model in models:
        lora_sd, _, _isv2 = load_state_dict(model, merge_dtype, "cpu")
        isv2 = isv2 or _isv2
        lora_sds.append(lora_sd)

    all_lora_module_names = set()
    for lora_sd in lora_sds:
        for key in lora_sd.keys():
            if 'lora_down' in key:
                lora_module_name = key[:key.rfind(".lora_down")]
                all_lora_module_names.add(lora_module_name)
    
    all_lora_module_names = sorted(list(all_lora_module_names))
    total_modules = len(all_lora_module_names)
    total_chunks = (total_modules + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Found {total_modules} unique modules to merge. Processing in {total_chunks} chunks of {CHUNK_SIZE}.")

    merged_lora_sd = {}

    with tqdm(total=total_modules, desc="Overall Progress") as pbar_overall:
        for i in range(0, total_modules, CHUNK_SIZE):
            chunk = all_lora_module_names[i:i + CHUNK_SIZE]
            
            pbar_overall.set_description(f"Processing Chunk {i//CHUNK_SIZE + 1}/{total_chunks}")

            merged_sd_chunk = {}
            original_shapes_chunk = {}

            for lora_module_name in chunk:
                merged_weight = None
                
                for j, lora_sd in enumerate(lora_sds):
                    ratio = ratios[j]
                    down_key = lora_module_name + '.lora_down.weight'
                    if down_key not in lora_sd:
                        continue

                    down_weight = lora_sd[down_key].to(device, non_blocking=True)
                    up_weight = lora_sd[lora_module_name + '.lora_up.weight'].to(device, non_blocking=True)
                    
                    network_dim = down_weight.size(0)
                    alpha = lora_sd.get(lora_module_name + '.alpha', torch.tensor(network_dim)).to(device, non_blocking=True)
                    scale = (alpha / network_dim) if network_dim else 0

                    conv2d = len(down_weight.size()) == 4
                    if not conv2d:
                        diff = (up_weight @ down_weight)
                    else:
                        diff = torch.nn.functional.conv2d(
                            down_weight.permute(1, 0, 2, 3), up_weight
                        ).permute(1, 0, 2, 3)

                    block_ratio = ratio[blockfromkey(down_key, LBLCOKS26, isv2)]
                    fugou = 1
                    if "same to Strength" in sets:
                        block_ratio, fugou = (block_ratio ** 0.5, 1) if block_ratio > 0 else (abs(block_ratio) ** 0.5, -1)
                    
                    if merged_weight is None:
                        merged_weight = (block_ratio * diff * scale * fugou)
                    else:
                        merged_weight += (block_ratio * diff * scale * fugou)

                if merged_weight is not None:
                    merged_sd_chunk[lora_module_name] = merged_weight
                    if len(merged_weight.shape) == 4:
                        original_shapes_chunk[lora_module_name] = merged_weight.shape

            with torch.no_grad():
                for lora_module_name, mat in merged_sd_chunk.items():
                    mat = mat.to(torch.float)
                
                    conv2d = lora_module_name in original_shapes_chunk
                    if conv2d:
                        out_dim, in_dim, k_h, k_w = original_shapes_chunk[lora_module_name]
                        mat = mat.reshape(out_dim, -1)

                    U, S, Vh = torch.linalg.svd(mat)

                    U = U[:, :new_rank]
                    S = S[:new_rank]
                    U = U @ torch.diag(S)
                    Vh = Vh[:new_rank, :]

                    dist = torch.cat([U.flatten(), Vh.flatten()])
                    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
                    low_val = -hi_val
                    U = U.clamp(low_val, hi_val)
                    Vh = Vh.clamp(low_val, hi_val)

                    new_up_weight = U
                    new_down_weight = Vh

                    if conv2d:
                        out_dim, in_dim, k_h, k_w = original_shapes_chunk[lora_module_name]
                        new_up_weight = new_up_weight.unsqueeze(2).unsqueeze(3)
                        new_down_weight = new_down_weight.view(new_rank, in_dim, k_h, k_w)

                    merged_lora_sd[lora_module_name + '.lora_up.weight'] = new_up_weight.to("cpu").contiguous()
                    merged_lora_sd[lora_module_name + '.lora_down.weight'] = new_down_weight.to("cpu").contiguous()
                    merged_lora_sd[lora_module_name + '.alpha'] = torch.tensor(float(new_rank))

                    pbar_overall.update(1)

            del merged_sd_chunk, original_shapes_chunk, chunk
            gc.collect()
            torch.cuda.empty_cache()

    print("LoRA merge process completed.")
    return merged_lora_sd

def extract_two(a,b,pa,pb,ps):
    base_alphas = {}                          # alpha for merged model
    base_dims = {}
    merged_sd = {}
    alphas = {}                             # alpha for current model
    dims = {}                               # dims for current model

    base_dims_a, base_alphas_a, dims, alphas_a = dimalpha(a, base_dims, base_alphas)
    base_dims_b, base_alphas_b, dims, alphas_b = dimalpha(b, base_dims, base_alphas)

    print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

    # merge
    print(f"merging...")
    for key in a.keys():
        if 'alpha' in key:
            continue

        lora_module_name = key[:key.rfind(".lora_")]

        base_alpha_a = base_alphas_a[lora_module_name]
        base_alpha_b = base_alphas_b[lora_module_name]
        alpha_a = alphas_a[lora_module_name]
        alpha_b = alphas_b[lora_module_name]

        scale_a = math.sqrt(alpha_a / base_alpha_a) 
        scale_b = math.sqrt(alpha_b / base_alpha_b)

        merged_sd[key] = extract_super(None,a[key] * scale_a,b[key] * scale_b,pa,pb,ps)

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    return merged_sd

def lycomerge(filename, ratios, calc_precision, device):
    merge_dtype = str_to_dtype(calc_precision)
    sd, metadata, isv2 = load_state_dict(filename, merge_dtype, device)

    if len(ratios) == 17:
      r0 = 1
      ratios = [ratios[0]] + [r0] + ratios[1:3]+ [r0] + ratios[3:5]+[r0] + ratios[5:7]+[r0,r0,r0] + [ratios[7]] + [r0,r0,r0] + ratios[8:]

    print("LyCORIS: " , ratios)

    keys_failed_to_match = []

    for lkey, weight in sd.items():
        ratio = 1
        picked = False
        if 'alpha' in lkey:
          continue

        fullkey = convert_diffusers_name_to_compvis(lkey,isv2)

        if "." not in fullkey:continue

        key, lora_key = fullkey.split(".", 1)

        for i,block in enumerate(LBLCOKS26):
            if block in key:
                ratio = ratios[i]
                picked = True
        if not picked: keys_failed_to_match.append(key)

        sd[lkey] = weight * math.sqrt(abs(float(ratio)))

        if "down" in lkey and ratio < 0:
          sd[key] = sd[key] * -1
        
    if len(keys_failed_to_match) > 0:
      print(keys_failed_to_match)
  
    return sd 

##############################################################
####### merge to checkpoint
def pluslora(lnames,loraratios,settings,output,model,save_precision,calc_precision,metasets,device):
    if model == []: return "ERROR: No model Selected"
    if lnames == "":return "ERROR: No LoRA Selected"
    import lora
    print("Plus LoRA start")
    add = ""

    temp = []
    for n in lnames.split(","):
        if ":" in n:
            temp.append(n.split(":"))
        else:
            temp[-1].append(n)
    
    lnames = temp

    loraratios=loraratios.splitlines()
    ldict ={}

    for l in loraratios:
        if ":" not in l or not any(l.count(",") == x - 1 for x in BLOCKNUMS) : continue
        ldict[l.split(":")[0].strip()]=l.split(":")[1]

    names, filenames, lweis = [], [], []

    for n in lnames:
        if len(n) ==2:
            ratio = [float(n[1])]*26
        elif len(n) ==3:
            if n[2].strip() in ldict:
                ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
                ratio = to26(ratio)
            else:ratio = [float(n[1])]*26
        elif len(n[2:]) in BLOCKNUMS:
            ratio = [float(x) for x in n[2:]]
            ratio = to26(ratio)
        else:ratio = [float(n[1])]*26
 
        c_lora = lora.available_loras.get(n[0], lora.available_lora_aliases.get(n[0],None)) 
        names.append(n[0])
        filenames.append(c_lora.filename)
        lweis.append(ratio)

    lora_families = []
    for filename in filenames:
        header = load_lora_header_or_state(filename)
        lora_families.append(anima_lora_family_from_state_dict(header))

    modeln=filenamecutter(model,True)
    dname = modeln
    for n in names:
      dname = dname + "+"+n

    checkpoint_info = sd_models.get_closet_checkpoint_match(model)

    if forge:
        revert_target = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
    print(f"Loading {model}")

    theta_0 = read_model_state_dict(checkpoint_info, device)
    dtype = qdtyper(theta_0)

    if dtype == "fp4" or dtype == "nf4":
        print(f"Changing dtype of {model} from {dtype} to float16")
        qkeys = list(theta_0.keys())
        q_dequantize(theta_0,dtype,device,torch.float16,False)

    base_is_anima = anima_is_checkpoint_state_dict(theta_0)
    if base_is_anima or any(family == "Anima" for family in lora_families):
        if not base_is_anima:
            return "ERROR: Anima LoRA can only be baked into an Anima checkpoint"
        if not all(family == "Anima" for family in lora_families):
            return "ERROR: Non-Anima LoRA cannot be baked into an Anima checkpoint"
        try:
            lweis = [parse_lora_ratio_spec(n, ldict, "Anima") for n in lnames]
            theta_0 = pluslora_anima(theta_0,filenames,lweis,calc_precision,device)
        except Exception as e:
            traceback.print_exc()
            return f"ERROR: {e}"

        settings.append(save_precision)
        settings.append("safetensors")
        result = savemodel(theta_0,dname,output,settings)

        lora.loaded_loras.clear()
        if hasattr(sd_models, "checkpoints_loaded"):
            sd_models.checkpoints_loaded.clear()
        if forge:
            from modules.sd_models import FakeInitialModel
            sd_models.unload_model_weights()
            sd_models.checkpoint_info = FakeInitialModel()
            load_model(revert_target, reload=True)

        del theta_0
        gc.collect()
        return result + add

    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0.keys()
    isv2 = "cond_stage_model.model.transformer.resblocks.0.attn.out_proj.weight" in theta_0.keys()
    isflux = any("double_block" in k for k in theta_0.keys())
    need_revert = prefixer(theta_0) if isflux else False

    if isflux:
        lweis = to61s(lweis)

    try:
        import networks
        is15 = True
    except:
        is15 = False

    keychanger = {}
    for key in theta_0.keys():
        skey = key.replace(".","_").replace("_weight","")
        if "conditioner_embedders_" in skey:
            keychanger[skey.split("conditioner_embedders_",1)[1]] = key
        else:
            if "wrapped" in skey:
                keychanger[skey.split("wrapped_",1)[1]] = key
            elif "clip_l" in skey or "t5xxl" in skey:
                keychanger[skey.replace("text_encoders_","")] = key
            elif "model_" in skey:
                keychanger[skey.split("model_",1)[1]] = key

    lowvramdealer() #web-uiのバグ対策

    if is15:
        if shared.sd_model is not None:
            orig_checkpoint = shared.sd_model.sd_checkpoint_info if hasattr(shared.sd_model, "sd_checkpoint_info") else None
        else:
            orig_checkpoint = None
        checkpoint_info = sd_models.get_closet_checkpoint_match(model)
        if orig_checkpoint != checkpoint_info:
            sd_models.reload_model_weights(info=checkpoint_info)
        
        theta_0 = newpluslora(theta_0,filenames,lweis,names, calc_precision, isxl,isv2,isflux, keychanger)
        
        if dtype == "fp4" or dtype == "nf4":
            print(f"Changing dtype of {model} from float16 to {dtype}")
            q_quantize(theta_0,dtype,device,False)
        
            failedkeys = []
            for key in theta_0:
                if key not in qkeys:
                    failedkeys.append(key)

            print(f"Key Check : {'OK' if failedkeys == [] else str(len(failedkeys)) + ' keys failed'}")

        if need_revert:
            prefixer(theta_0, True)

        if orig_checkpoint:
            sd_models.reload_model_weights(info=orig_checkpoint)
    else:
        theta_0 = oldpluslora(theta_0,filenames,lweis,names, calc_precision,isxl,isv2, keychanger, device)

    #usemodelgen(theta_0,model)
    settings.append(save_precision)
    settings.append("safetensors")
    result = savemodel(theta_0,dname,output,settings)

    lora.loaded_loras.clear()
    if hasattr(sd_models, "checkpoints_loaded"):
        sd_models.checkpoints_loaded.clear()
    if forge:
        from modules.sd_models import FakeInitialModel
        sd_models.unload_model_weights()
        sd_models.checkpoint_info = FakeInitialModel()
        load_model(revert_target, reload=True)

    del theta_0
    gc.collect()
    return result + add

def oldpluslora(theta_0,filenames,lweis,names, calc_precision,isxl,isv2, keychanger, device):
    for name,filename, lwei in zip(names,filenames, lweis):
        print(f"loading: {name}")
        lora_sd, metadata, isv2 = load_state_dict(filename, torch.float, device)

        print(f"merging..." ,lwei)
        for key in lora_sd.keys():
            ratio = 1
            fullkey = convert_diffusers_name_to_compvis(key,isv2)

            msd_key, _ = fullkey.split(".", 1)
            if isxl:
                if "lora_unet" in msd_key:
                    msd_key = msd_key.replace("lora_unet", "diffusion_model")
                elif "lora_te1_text_model" in msd_key:
                    msd_key = msd_key.replace("lora_te1_text_model", "0_transformer_text_model")

            for i,block in enumerate(LBLCOKS26):
                if block in fullkey or block in msd_key:
                    ratio = lwei[i]

            if "lora_down" in key:
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[:key.index("lora_down")] + 'alpha'

                # print(f"apply {key} to {module}")

                down_weight = lora_sd[key].to(device="cpu")
                up_weight = lora_sd[up_key].to(device="cpu")

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim
                # W <- W + U * D
                weight = theta_0[keychanger[msd_key]].to(device="cpu")

                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale

                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # print(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale
                    
                theta_0[keychanger[msd_key]] = torch.nn.Parameter(weight)
    return theta_0

def newpluslora(theta_0,filenames,lweis,names, calc_precision,isxl,isv2,isflux, keychanger):
    if nets is None:
        raise RuntimeError("SuperMerger checkpoint LoRA merge needs the A1111 networks shim, which is unavailable on this Forge Neo runtime.")
    nets.load_networks(names, [1]* len(names),[1]* len(names), None, isxl, isv2)

    for l, loaded in enumerate(nets.loaded_networks):
        for n, name in enumerate(names):
            changed = False
            if name == loaded.name:
                lbw(nets.loaded_networks[l],to26(lweis[n]),isv2,isflux)
                changed = True
            if not changed: "ERROR: {name}weight is not changed"

    errormodules = []
    for net in nets.loaded_networks:
        net.dyn_dim = None
        for name,module in tqdm(net.modules.items(), desc=f"{net.name}"):
            fullkey = convert_diffusers_name_to_compvis(name,isv2)
            msd_key = fullkey.split(".")[0]

            if isxl:
                if "lora_unet" in msd_key:
                    msd_key = msd_key.replace("lora_unet", "diffusion_model")
                elif "lora_te1_text_model" in msd_key:
                    msd_key = msd_key.replace("lora_te1_text_model", "0_transformer_text_model")

            qvk = ["_q_proj","_k_proj","_v_proj","_out_proj"]

            if msd_key in keychanger.keys():
                wkey = keychanger[msd_key]
                bkey = wkey.replace("weight","bias")
                if bkey in theta_0.keys():
                    theta_0[wkey], theta_0[bkey]= plusweights(theta_0[wkey], module, bias = theta_0[bkey])
                else:
                    theta_0[wkey], _ = plusweights(theta_0[wkey] ,module)

            else:
                if any(x in name for x in qvk):
                    for x in qvk:
                        if x in name:
                            inkey,outkey = name.replace(x,"") + "_in_proj" ,name.replace(x,"") + "_out_proj"
                    bkey = keychanger[outkey].replace("weight","bias")
                    if bkey in theta_0.keys():
                        theta_0[keychanger[inkey]] ,theta_0[keychanger[outkey]], theta_0[bkey]= plusweightsqvk(theta_0[keychanger[inkey]],theta_0[keychanger[outkey]], name ,module, net, bias = theta_0[bkey])
                    else:
                        theta_0[keychanger[inkey]] ,theta_0[keychanger[outkey]], _= plusweightsqvk(theta_0[keychanger[inkey]],theta_0[keychanger[outkey]], name ,module, net)
                else:
                    errormodules.append(msd_key)
                    
        if errormodules != []:
            print(f"Unmerged modules in {net.name} : {errormodules}")
        gc.collect()
    return theta_0

def plusweights(weight, module, bias = None):
    with torch.no_grad():
        if weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e5m2:  # Float8 の場合
            orig_dtype = weight.dtype
            weight = weight.to(torch.float32)  # Float32 に変換
        else:
            orig_dtype = None
        updown = module.calc_updown(weight.to(dtype=torch.float32))
        if len(weight.shape) == 4 and weight.shape[1] == 9:
            # inpainting model. zero pad updown to make channel[1]  4 to 9
            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))
        if type(updown) == tuple:
            updown, ex_bias = updown
            if ex_bias is not None and bias is not None:
                bias += ex_bias
        weight += updown.to(weight.dtype)
    return weight if orig_dtype is None else weight.to(orig_dtype), bias

def plusweightsqvk(inweight, outweight, network_layer_name, module ,net,bias = None):
    with torch.no_grad():
        module_q = net.modules.get(network_layer_name + "_q_proj", None)
        module_k = net.modules.get(network_layer_name + "_k_proj", None)
        module_v = net.modules.get(network_layer_name + "_v_proj", None)
        module_out = net.modules.get(network_layer_name + "_out_proj", None)

        if module_q and module_k and module_v and module_out:
            with torch.no_grad():
                updown_q = module_q.calc_updown(inweight)
                updown_k = module_k.calc_updown(inweight)
                updown_v = module_v.calc_updown(inweight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                updown_out = module_out.calc_updown(outweight)
                if type(updown_out) is tuple:
                    updown_out,ex_bias = updown_out

                inweight += updown_qkv
                outweight += updown_out
                if bias is not None and ex_bias is not None:
                    bias += ex_bias

    return inweight,outweight,bias

def lbw(lora,lwei,isv2,isflux=False):
    errormodules = []

    blocks = LBLCOKS26
    if isv2:
        blocks[0] = V2ENCODER

    for key in lora.modules.keys():
        ratio = 1
        picked = False

        if isflux:
            block = get_flux_blocks(key)
            if block in BLOCKIDFLUX:
                ratio = lwei[BLOCKIDFLUX.index(block)]
                picked = True

        else:
            for i,block in enumerate(blocks):
                if block in key:
                    if i == 26 or i == 27: i=0
                    ratio = lwei[i]
                    picked = True

        if not picked:
            errormodules.append(key)

        ltype = type(lora.modules[key]).__name__

        set = False
        if ltype in LORAANDSOON.keys():
            setattr(lora.modules[key],LORAANDSOON[ltype],torch.nn.Parameter(getattr(lora.modules[key],LORAANDSOON[ltype]) * ratio))
            #print(ltype)
            set = True
        else:
            if hasattr(lora.modules[key],"up_model"):
                lora.modules[key].up_model.weight= torch.nn.Parameter(lora.modules[key].up_model.weight *ratio)
                #print("LoRA using LoCON")
                set = True
            else:
                lora.modules[key].up.weight= torch.nn.Parameter(lora.modules[key].up.weight *ratio)
                #print("LoRA")
                set = True
        if not set : 
            print("unkwon LoRA")

    if errormodules:
        print("unchanged modules in lbw:", errormodules)
    else:
        print(f"{lora.name}: Successfully set the ratio {lwei} ")

    return lora

LORAANDSOON = {
    "LoraHadaModule" : "w1a",
    "LycoHadaModule" : "w1a",
    "NetworkModuleHada": "w1a",
    "FullModule" : "weight",
    "NetworkModuleFull": "weight",
    "IA3Module" : "w",
    "NetworkModuleIa3" : "w",
    "LoraKronModule" : "w1",
    "LycoKronModule" : "w1",
    "NetworkModuleLokr": "w1",
    "NetworkModuleNorm": "w_norm",
}

def save_to_file(file_name, model, state_dict, dtype, metadata):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata=metadata)
    else:
        torch.save(model, file_name)

CLAMP_QUANTILE = 0.99

def str_to_dtype(p):
  if p == 'float':
    return torch.float
  if p == 'fp16':
    return torch.float16
  if p == 'bf16':
    return torch.bfloat16
  return None


def get_safetensors_header(filename):
    import json
    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        if metadata_len > 2 and json_start in (b'{"', b"{'"):
            json_data = json_start + file.read(metadata_len-2)
            return json.loads(json_data)

        # invalid safetensors
        return {}

def load_state_header(file_name, dtype):
  """load safetensors header if available"""
  if os.path.splitext(file_name)[1] == '.safetensors':
    sd = get_safetensors_header(file_name)
  else:
    sd = torch.load(file_name, map_location='cpu')
  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)
  return sd

def load_state_dict(file_name, dtype, device = "cpu"):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name,device=device)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location=device)
        metadata = {}

    isv2 = False

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype = dtype, device = device)
            if "resblocks" in key:
                isv2 = True
    
    if isv2: print("SD2.X")

    return sd, metadata, isv2

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata

def dimgetter(filename, device = "cpu"):
    lora_sd = load_state_header(filename, torch.float)
    alpha = None
    dim = None
    ltype = None

    if anima_is_lora_state_dict(lora_sd):
        lora_sd = normalize_anima_lora_state_dict(lora_sd)
        for key, value in lora_sd.items():
            if key.endswith(".lora_down.weight"):
                dim = tensor_rank(value)
                break
        return dim if dim else "unknown", "LoRA", "Anima"

    if "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight" in lora_sd.keys():
      ltype = "LoCon"
      if type(lora_sd["lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight"]) is dict:
          lora_sd, _, _ = load_state_dict(filename, torch.float, device)
      _, _, dim, _ = dimalpha(lora_sd)

    if "lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight" in lora_sd.keys():
        sdx = "XL"
        if type(lora_sd["lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight"]) is dict:
            lora_sd, _, _ = load_state_dict(filename, torch.float, device)
        _, _, dim, _ = dimalpha(lora_sd)
    elif "lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.hada_w1_a" in lora_sd.keys():
        sdx = "XL"
    elif any("single_blocks" in key for key in lora_sd.keys()):
        sdx = "Flux"
    else:
        sdx = "1.X/2.X"

    if isinstance(dim, dict):
        dim = d2l(dim)

    for key, value in lora_sd.items():
  
        if alpha is None and 'alpha' in key:
            alpha = value
        if dim is None and 'lora_down' in key:
            if type(value) == torch.Tensor and len(value.size()) == 2:
                dim = value.size()[0]
            elif type(value) == dict:
                dim = value.get("shape",[0,0])[0]
        if "hada_" in key:
            dim,ltype = "LyCORIS","LyCORIS"
        if alpha is not None and dim is not None:
            break
    if alpha is None:
        alpha = dim
    if ltype == None:ltype = "LoRA"
    if dim :
      return dim, ltype, sdx
    else:
      return "unknown","unknown","unknown"
    
def d2l(dimdict):
    out = []
    for v in dimdict.values():
        if v not in out:
            out.append(v)
    return out[0] if len(out) == 1 else out

def blockfromkey(key,keylist,isv2 = False):
    fullkey = convert_diffusers_name_to_compvis(key,isv2)

    if "lora_unet" in fullkey:
        fullkey = fullkey.replace("lora_unet", "diffusion_model")
    elif "lora_te1_text_model" in fullkey:
        fullkey = fullkey.replace("lora_te1_text_model", "0_transformer_text_model")

    if "1_model_transformer_resblocks_" in fullkey:return 0

    for i,n in enumerate(keylist):
        if n in fullkey: return i

    print(f"ERROR:Block is not deteced:{fullkey}")
    return 0

def dimalpha(lora_sd, base_dims={}, base_alphas={}):
    alphas = {}                             # alpha for current model
    dims = {}                               # dims for current model
    for key in lora_sd.keys():
        if 'alpha' in key:
            lora_module_name = key[:key.rfind(".alpha")]
            alpha = float(lora_sd[key].detach().cpu().numpy()) 
            alphas[lora_module_name] = alpha
            if lora_module_name not in base_alphas:
                base_alphas[lora_module_name] = alpha
        elif "lora_down" in key:
            lora_module_name = key[:key.rfind(".lora_down")]
            dim = lora_sd[key].size()[0]
            dims[lora_module_name] = dim
            if lora_module_name not in base_dims:
                base_dims[lora_module_name] = dim

    for lora_module_name in dims.keys():
        if lora_module_name not in alphas:
            alpha = dims[lora_module_name]
            alphas[lora_module_name] = alpha
            if lora_module_name not in base_alphas:
                base_alphas[lora_module_name] = alpha
    return base_dims, base_alphas, dims, alphas


def fullpathfromname(name):
    if hash == "" or hash ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    return checkpoint_info.filename

def makeloraname(model_a,model_b):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    return "lora_"+model_a+"-"+model_b

V2ENCODER = "resblocks"

LBLCOKS26=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders",
"transformer_resblocks"]

###########################################################
##### metadata

def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash

def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]

def prepare_merge_metadata( ratio, blocks, fromLora ):
    """
    メタデータに ratio, blocks などの情報を付加しておく

    Parameters
    ----
    ratio : string
        name:ratio:blocks の ratio 部分
    blocks : string
        name:ratio:bloks の blocks 部分(ラベルではなくて実パラメータ)
    fromLora : NetworkOnDisk
        マージ対象のLoRA
    
    Returns
    ----
    dict[str, str]
        メタデータ
    """
    meta = fromLora.metadata
    meta["sshs_ratio"] = str.strip( ratio )
    meta["sshs_blocks"] = str.strip( blocks )
    meta["ss_output_name"] = str.strip( fromLora.name )

    return meta

BASE_METADATA = [
    "sshs_ratio", "sshs_blocks", "ss_output_name",
    "sshs_model_hash", "sshs_legacy_hash",
    "ss_network_module",
    "ss_network_alpha", "ss_network_dim",
    "ss_mixed_precision", "ss_v2",
    "ss_training_comment",
    "ss_sd_model_name", "ss_new_sd_model_hash",
    "ss_clip_skip",
    "ss_base_model_version"
]

MINIMUM_METADATA = [
    "ss_network_module","ss_network_alpha", "ss_network_dim","ss_v2","ss_sd_model_name", "ss_base_model_version"
]

def create_merge_metadata( sd, lmetas, lname, lprecision, metasets ):
    """
    LoRAマージ後のメタデータを作成する

    Parameters
    ----
    sd : NetworkOnDisk
        マージ後のLoRA
    lmetas : dict[str, str]
        マージされるLoRAのメタデータ
    lname : str
        マージ後のLoRA名
    lprecision : str
        save precision の値
    mergeAll : bool
        メタデータの残し方。ただしタグ情報はディレクトリ名が後勝ちでマージします
        True 全メタデータを残す。単マージの場合はTrue固定
        False 一部のメタデータのみ残す
    
    Returns
    ----
    dict[str, str]
        メタデータ
    """

    metadata = {}
    networkModule = None

    if "first" in metasets:
        # 単なるweightマージならそのままコピー
        metadata = lmetas[0]
    elif "new" in metasets:
        for key in MINIMUM_METADATA:
            if key in lmetas[0].keys():
                metadata[key] = lmetas[0][key]
        
    else:
        # 複数マージの場合はマージしたタグと主要メタデータを保存
        metadata = lmetas[0]
        tags = {}
        for i, lmeta in enumerate( lmetas ):
            meta = {}
            metadata[ f"sshs_cp{i}" ] = json.dumps( lmeta )

            # 最初の network_module を保持
            if networkModule is None and "ss_network_module" in lmeta:
                networkModule = lmeta["ss_network_module"]

            # タグをマージ
            if "merge" in metasets:
                if "ss_tag_frequency" in lmeta:
                    ldict = lmeta["ss_tag_frequency"]
                    if "ss_tag_frequency" in metadata:
                        mdict = metadata["ss_tag_frequency"]
                        if type(ldict) is dict and type(mdict) is dict:
                            for key in ldict:
                                if key not in mdict:
                                    mdict[key] = ldict[key]

    # network_moduleからLoRA種別判定する場合が多いため、最初に見つけたものにする
    if networkModule is not None:
        metadata["ss_network_module"] = networkModule

    # output名とprecision、dimは変更された可能性がある
    if "without" not in metasets:
        metadata["ss_output_name"] = lname
    else:
        if "ss_output_name" in metadata:
            del metadata["ss_output_name"]
    metadata["ss_mixed_precision"] = lprecision

    # metadataで保存できる形式に変換
    for key in metadata:
        if type(metadata[key] ) is not str:
            metadata[key] = json.dumps( metadata[key] )
    # データ変更によりhashが変わるので計算
    model_hash, legacy_hash = precalculate_safetensors_hashes( sd, metadata )
    metadata[ "sshs_model_hash" ] = model_hash
    metadata[ "sshs_legacy_hash" ] = legacy_hash

    return metadata


##############################################################
####### Get loranames from prompt
def frompromptf(*args):
    outst = []
    outss = []
    prompt = args[1]
    names, multis, lbws = loradealer(prompt, "", "")
    for name, multi, lbw in zip(names, multis, lbws):
        nml = [name,str(multi),lbw] if lbw is not None else [name,str(multi)]
        outst.append(":".join(nml))
        choice = lora_choice_for_name(name)
        if choice is not None:
            outss.append(choice)
    global pchanged
    pchanged = True
    return lora_dropdown_update(outss),",".join(outst), True

def loradealer(prompts,lratios,elementals):
    _, extra_network_data = extra_networks.parse_prompts([prompts])
    moduletypes = extra_network_data.keys()

    outnames = []
    outmultis = []
    outlbws = []

    for ltype in moduletypes:
        lorans = []
        lorars = []
        loraps = []
        multipliers = []
        elements = []
        if not (ltype == "lora" or ltype == "lyco") : continue
        for called in extra_network_data[ltype]:
            multiple = float(syntaxdealer(called.items,"unet=","te=",1))
            multipliers.append(multiple)
            lorans.append(called.items[0])
            loraps.append(syntaxdealer(called.items,"lbw=",None,2))

        if len(lorans) > 0:
            outnames.extend(lorans)
            outmultis.extend(multipliers)
            outlbws.extend(loraps)

    return outnames, outmultis, outlbws

def syntaxdealer(items,type1,type2,index): #type "unet=", "x=", "lwbe=" 
    target = [type1,type2] if type2 is not None else [type1]
    for t in target:
        for item in items:
            if t in item:
                return item.replace(t,"")
    if index > len(items) - 1 :return None
    return items[index] if "@" not in items[index] else 1

##############################################################
####### Extract lora from checkpoints args
class Kohya_extract_args:
    def __init__(
        self,
        v2=False,
        v_parameterization=None,
        sdxl=False,
        save_precision=None,
        model_org=None,
        model_tuned=None,
        save_to=None,
        dim=4,
        conv_dim=None,
        device=None,
        no_metadata=False,
        alpha = 1,
        beta = 1
    ):
        self.v2 = v2
        self.v_parameterization = v_parameterization
        self.sdxl = sdxl
        self.save_precision = save_precision
        self.model_org = model_org
        self.model_tuned = model_tuned
        self.save_to = save_to
        self.dim = dim
        self.conv_dim = conv_dim
        self.device = device
        self.no_metadata = no_metadata
        self.alpha = alpha
        self.beta = beta

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    #for flux
    if match(m, r"lora_unet_double_blocks_(\d+)_(img|txt)_(attn|mlp|mod)_(proj|qkv|lin|\d+)(.*)"):
        block_type = m[1]  # img or txt
        module_type = m[2]  # attn, mlp, mod
        specific_module = m[3]  # proj, qkv, lin, or numeric index

        # Create suffix for specific module types
        if module_type == "attn":
            if specific_module == "proj":
                suffix = "proj.weight"
            elif specific_module == "qkv":
                suffix = "qkv.weight"
            else:
                suffix = f"norm.{specific_module}_norm.scale"
        elif module_type == "mlp":
            suffix = f"{specific_module}.weight"
        elif module_type == "mod":
            suffix = f"lin.weight"
        else:
            suffix = specific_module

        return f"model.diffusion_model.double_blocks.{m[0]}.{block_type}_{module_type}.{suffix}"

    if match(m, r"lora_unet_single_blocks_(\d+)_(linear\d+|modulation_lin)(.*)"):
        block_index = m[0]  # single block index
        module_name = m[1]  # linear1, linear2, or modulation_lin

        # Create suffix for module types
        if "linear" in module_name:
            suffix = f"{module_name}.weight"
        elif module_name == "modulation_lin":
            suffix = "modulation.lin.weight"
        else:
            suffix = module_name

        return f"model.diffusion_model.single_blocks.{block_index}.{suffix}"

    return key

def lowvramdealer():
    try:
        from modules import lowvram
        lowvram.module_in_gpu = None #web-uiのバグ対策
    except:
        pass

def get_flux_blocks(key):
    if "vae" in key:
        return "VAE"
    if "t5xxl" in key:
        return "T5"
    if "clip_l" in key:
        return "CLIP"
    
    match = re.search(r'\_(\d+)\_', key)
    if "double_blocks" in key:
        return f"D{match.group(1).zfill(2) }"
    if "single_blocks" in key:
        return f"S{match.group(1).zfill(2) }"
    if "_in" in key:
        return "IN"
    if "final_layer" in key:
        return "OUT"
    return "Not Merge"

def read_model_state_dict(checkpoint_info, device):
    if forge:
        from backend.utils import load_torch_file
        load_model(checkpoint_info)
        return load_torch_file(checkpoint_info.filename,device=CUDA if "cuda" in device else CPU)
    else:
        return sd_models.read_state_dict(checkpoint_info.filename,map_location=device)
    
def load_model(checkpoint_info, reload = False):
    if forge:
        from modules.sd_models import forge_model_reload, model_data
        from modules_forge.main_entry import forge_unet_storage_dtype_options
        unet_storage_dtype, _ = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))
        forge_model_params = dict(
            checkpoint_info=checkpoint_info,
            additional_modules=shared.opts.forge_additional_modules,
            unet_storage_dtype=unet_storage_dtype
        )
        if reload:
            model_data.forge_hash = None
        model_data.forge_loading_parameters = forge_model_params
        forge_model_reload()
    else:
        sd_models.load_model(checkpoint_info)
