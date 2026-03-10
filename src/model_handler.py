'''
This file has two jobs:
1. Load a model (Llama-2-7B or OLMo-2-7B) and its tokenizer from HuggingFace
2. Patch the model's attention so the LAHIS head mask works — this is the part we had to write ourselves

Supported model names for load_model():
  "llama2"  → meta-llama/Llama-2-7b-hf
  "olmo2"   → allenai/OLMo-2-7B  (or any HuggingFace path via model_name="hf:some/path")
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
import torch
import torch.nn.functional as F
import math
try:
    from nethook import Trace, TraceDict
except ImportError:
    Trace = TraceDict = None  # only needed for trace_model / output_layer_logits
import dotenv
import os
import json

# Model path — HuggingFace model IDs
AYA_NAME     = "CohereLabs/aya-23-8B"
MISTRAL_NAME = "mistralai/Mistral-7B-v0.1"
LLAMA3_NAME  = "meta-llama/Llama-3.2-3B"
LLAMA2_NAME  = "meta-llama/Llama-2-7b-hf"
OLMO2_NAME   = "allenai/OLMo-2-1124-7B"   # ← base OLMo-2-7B (November 2024 release)

# TODO Input your local path of LLMs
AYA_NAME_LOCAL     = "../../autodl-tmp/aya-23-8B"
MISTRAL_NAME_LOCAL = "../../autodl-tmp/Mistral-7B-v0.1"
LLAMA3_NAME_LOCAL  = "../../autodl-tmp/Llama-3.2-3B"
LLAMA2_NAME_LOCAL  = "../../autodl-tmp/Llama-2-7b-hf"
OLMO2_NAME_LOCAL   = "../../autodl-tmp/OLMo-2-1124-7B"   # local path if downloaded


# Get access token
dotenv.load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

#These 6 functions came with the original LAHIS repo. We don't call most of them in our pipeline 
def load_json(json_path):
    with open(json_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)

    return data


def save_json(json_path, data):
    with open(json_path, "w", encoding='utf-8') as fp:
        json.dump(data, fp, indent=4)

    return


def generate_with_ori_model(model, tokenizer, input_ids_batch, max_new_tokens=10, head_mask=None, head_importance=None):
    with torch.no_grad():
        output_dict_ori = model.generate(input_ids_batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                                         return_dict_in_generate=True, output_logits=True,
                                         head_mask=head_mask)
        output = tokenizer.batch_decode(output_dict_ori.sequences, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)

    # print(f"ori output: {output}")

    return output


def load_layer_param(model, target_layer, save_path):
    layer_state_dict = torch.load(save_path, map_location=model.device)
    model_dict = model.state_dict()

    for name, param in layer_state_dict.items():
        if name in model_dict:
            model_dict[name].copy_(param)
        else:
            print(f"Not found {name} in model.")

    print(f"Load layer {target_layer} weights from {save_path}")


def save_layer_param(model, target_layer, save_path):
    layer_state_dict = {
        name: param for name, param in model.state_dict().items()
        if f".{target_layer}." in name
    }

    torch.save(layer_state_dict, save_path)
    print(f"Save layer {target_layer} weights to {save_path}")


def get_model_inputs(tokenizer, text, device):
    # inputs: input_ids, attention_mask, offset_mapping
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, padding="longest")
    inputs.to(device)

    return inputs

# these use nethook (that library we made optional) to intercept the hidden states at each layer mid-computation. 
# Used for probing what each layer "knows" at inference time. We don't use these for now
def trace_model(model, input_ids, head_mask, token_num=1, module_prefix="model.layers.", target=""):
    hook_layers = [f'{module_prefix}{l}{target}' for l in range(model.config.num_hidden_layers)]

    with TraceDict(model, layers=hook_layers, retain_output=True) as res:
        model(input_ids, head_mask=head_mask)

    if len(res[hook_layers[0]].output[0].shape) == 3:
        output_res = [res[hook].output[0][0][-token_num:] for hook in hook_layers]
    elif len(res[hook_layers[0]].output[0].shape) == 2:
        output_res = [res[hook].output[0][-token_num:] for hook in hook_layers]
    else:
        raise ValueError("Other res[hook_layers[0]].output[0].shape")

    return output_res


def output_layer_logits(model, tokenizer, input_ids, head_mask, token_num=1):
    hs_res = trace_model(model, input_ids, head_mask, token_num, module_prefix="model.layers.", target="")

    tokens_list = []
    possis_list = []
    for l in range(model.config.num_hidden_layers):
        logits_l = model.lm_head(model.model.norm(hs_res[l]))
        logit_softmax_l = torch.softmax(logits_l, dim=-1)
        top_indices = torch.topk(logits_l, 1, dim=-1, sorted=True).indices

        decoded_tokens = [tokenizer.decode(top_indices[i], skip_special_tokens=False) for i in range(token_num)]
        tokens_list.append(decoded_tokens)
        token_possis = [logit_softmax_l[i][top_indices[i]].item() for i in range(token_num)]
        possis_list.append(token_possis)

        print(f"layer {l}: {decoded_tokens}")

    return tokens_list, possis_list


# ---------------------------------------------------------------------------
# Llama-2 head-mask patch
#
# HF transformers' LlamaAttention does not propagate head_mask down to the
# per-head output tensor (attn_output) by default — it is either ignored or
# only applied to attention logits/weights, which differs from the LAHIS paper
# formulation (Eq. 8 / README snippet).
#
# patch_llama_for_head_mask() replaces each LlamaAttention.forward with an
# implementation that multiplies attn_output (shape [B, H, T, d_head])
# by the mask BEFORE the output projection, exactly as the paper describes:
#
#   attn_output = attn_output * layer_head_mask.view(1, -1, 1, 1)
#
# Usage:
#   patch_llama_for_head_mask(model)          # call once after load_model
#   set_lahis_head_mask(model, head_mask)      # before every forward pass
#   clear_lahis_head_mask(model)              # to restore unmasked behaviour
# ---------------------------------------------------------------------------

def patch_llama_for_head_mask(model):
    """
    Monkey-patch all LlamaAttention layers so that setting
      layer.self_attn._lahis_mask = tensor([num_heads])
    applies a per-head scale to attn_output before o_proj.

    Reads all shape constants from model.config so the patch is robust
    against transformers version differences in attribute naming.
    """
    try:
        from transformers.models.llama.modeling_llama import (
            repeat_kv,
            apply_rotary_pos_emb,
        )
    except ImportError as e:
        raise ImportError(
            "transformers Llama utilities not found. "
            "Install transformers >= 4.35: pip install transformers>=4.35"
        ) from e

    # Read all scalar config values once from model.config — avoids
    # breakage when transformers renames attributes on the attention module.
    cfg = model.config
    num_heads     = cfg.num_attention_heads
    num_kv_heads  = cfg.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    hidden_size   = cfg.hidden_size
    head_dim      = getattr(cfg, "head_dim", hidden_size // num_heads)
    attn_dropout  = getattr(cfg, "attention_dropout", 0.0)
    scale         = math.sqrt(head_dim)

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._lahis_mask = None  # initialise mask slot to no-op

        def _make_forward(attn_module, l_idx):
            def _forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                query_states = attn_module.q_proj(hidden_states)
                key_states   = attn_module.k_proj(hidden_states)
                value_states = attn_module.v_proj(hidden_states)

                query_states = query_states.view(
                    bsz, q_len, num_heads, head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, num_kv_heads, head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, num_kv_heads, head_dim
                ).transpose(1, 2)

                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    cos, sin = attn_module.rotary_emb(value_states, position_ids)

                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos}
                    if cache_position is not None:
                        cache_kwargs["cache_position"] = cache_position
                    layer_id = getattr(attn_module, "layer_idx", l_idx)
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, layer_id, cache_kwargs
                    )

                key_states   = repeat_kv(key_states,   num_kv_groups)
                value_states = repeat_kv(value_states, num_kv_groups)

                attn_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) / scale

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                attn_weights = F.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_weights = F.dropout(
                    attn_weights,
                    p=attn_dropout,
                    training=attn_module.training,
                )

                # [bsz, num_heads, q_len, head_dim]
                attn_output = torch.matmul(attn_weights, value_states)

                # --- LAHIS head mask application (paper Eq. 8 / README) ---
                if attn_module._lahis_mask is not None:
                    mask = attn_module._lahis_mask.to(
                        device=attn_output.device, dtype=attn_output.dtype
                    )
                    attn_output = attn_output * mask.view(1, -1, 1, 1)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, (attn_weights if output_attentions else None)

            return _forward

        # Replace instance attribute (shadows class method without touching it)
        attn.forward = _make_forward(attn, layer_idx)

    return model


# ---------------------------------------------------------------------------
# OLMo-2 head-mask patch
#
# OLMo-2's attention (Olmo2Attention) is very similar to Llama-2 but has
# two extra steps that Llama-2 does NOT have:
#
#   1. q_norm / k_norm — RMSNorm applied per-head to queries and keys
#      AFTER projection and reshape, BEFORE rotary embeddings.
#      This stabilises training. We must include these or the forward
#      pass produces wrong results.
#
#   2. Slightly different reshape at the end — OLMo-2 uses
#      attn_output.reshape(*input_shape, -1) where input_shape = (bsz, q_len)
#      instead of Llama-2's explicit .reshape(bsz, q_len, hidden_size).
#      Both do the same thing, just written differently.
#
# Everything else (LAHIS mask application, o_proj, return values) is
# identical to the Llama-2 patch.
# ---------------------------------------------------------------------------

def patch_olmo2_for_head_mask(model):
    """
    Monkey-patch all Olmo2Attention layers so that setting
      layer.self_attn._lahis_mask = tensor([num_heads])
    applies a per-head scale to attn_output before o_proj.

    Differences from patch_llama_for_head_mask:
      - imports from transformers.models.olmo2 instead of llama
      - applies q_norm and k_norm after reshape (OLMo-2 specific)
      - uses attn_output.reshape(*input_shape, -1) at the end
    """
    try:
        from transformers.models.olmo2.modeling_olmo2 import (
            repeat_kv,           # same helper as Llama: repeats K/V heads for GQA
            apply_rotary_pos_emb # same rotary position embedding helper
        )
    except ImportError as e:
        raise ImportError(
            "transformers OLMo-2 utilities not found. "
            "Install transformers >= 4.47: pip install -U transformers"
        ) from e

    # Read architecture constants from config — same approach as Llama patch
    cfg           = model.config
    num_heads     = cfg.num_attention_heads   # 32 for OLMo-2-7B
    num_kv_heads  = cfg.num_key_value_heads   # 32 for OLMo-2-7B (no GQA)
    num_kv_groups = num_heads // num_kv_heads # = 1 (each Q head has its own KV)
    hidden_size   = cfg.hidden_size           # 4096
    head_dim      = hidden_size // num_heads  # 128
    attn_dropout  = getattr(cfg, "attention_dropout", 0.0)
    scale         = math.sqrt(head_dim)       # scaling factor for attention scores

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._lahis_mask = None  

        def _make_forward(attn_module, l_idx):
            def _forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()
                # input_shape used later for the final reshape
                input_shape = (bsz, q_len)

                # ── Step 1: Project hidden states to Q, K, V ──────────────
                # Same as Llama-2: three linear projections
                query_states = attn_module.q_proj(hidden_states)
                key_states   = attn_module.k_proj(hidden_states)
                value_states = attn_module.v_proj(hidden_states)

                # ── Step 2: q_norm and k_norm — OLMo-2 ONLY ───────────────
                # Applied BEFORE reshape. In OLMo-2-1124-7B, q_norm.weight has
                # size num_heads * head_dim = 4096, so it operates on the flat
                # [bsz, q_len, 4096] projection output. Applying it after
                # reshape (when last dim = 128) causes a shape mismatch error.
                query_states = attn_module.q_norm(query_states)
                key_states   = attn_module.k_norm(key_states)

                # ── Step 3: Reshape from flat to per-head ──────────────────
                # [bsz, q_len, num_heads * head_dim] → [bsz, num_heads, q_len, head_dim]
                query_states = query_states.view(
                    bsz, q_len, num_heads, head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, num_kv_heads, head_dim
                ).transpose(1, 2)
                value_states = value_states.view(
                    bsz, q_len, num_kv_heads, head_dim
                ).transpose(1, 2)

                # ── Step 4: Rotary position embeddings ────────────────────
                # Encodes the position of each token into Q and K
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                else:
                    cos, sin = attn_module.rotary_emb(value_states, position_ids)

                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                # ── Step 5: KV cache update (for generation) ──────────────
                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos}
                    if cache_position is not None:
                        cache_kwargs["cache_position"] = cache_position
                    layer_id = getattr(attn_module, "layer_idx", l_idx)
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, layer_id, cache_kwargs
                    )

                # ── Step 6: Expand K/V for grouped-query attention ─────────
                # OLMo-2-7B has num_kv_groups=1 so repeat_kv does nothing,
                # but we keep it for correctness if config changes
                key_states   = repeat_kv(key_states,   num_kv_groups)
                value_states = repeat_kv(value_states, num_kv_groups)

                # ── Step 7: Attention scores ───────────────────────────────
                # Q × K^T / sqrt(head_dim) → softmax → dropout
                attn_weights = torch.matmul(
                    query_states, key_states.transpose(2, 3)
                ) / scale

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                attn_weights = F.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
                attn_weights = F.dropout(
                    attn_weights, p=attn_dropout, training=attn_module.training
                )

                # ── Step 8: Weighted sum of values ─────────────────────────
                # [bsz, num_heads, q_len, head_dim]
                attn_output = torch.matmul(attn_weights, value_states)

                # ── Step 9: LAHIS head mask — identical to Llama-2 ─────────
                # If a mask is set, multiply each head's output by its scalar.
                # mask shape [32] → broadcast to [1, 32, 1, 1] so it scales
                # every token and every dimension of each head independently.
                if attn_module._lahis_mask is not None:
                    mask = attn_module._lahis_mask.to(
                        device=attn_output.device, dtype=attn_output.dtype
                    )
                    attn_output = attn_output * mask.view(1, -1, 1, 1)

                # ── Step 10: Flatten heads back and apply output projection ─
                # OLMo-2 uses reshape(*input_shape, -1) — same result as
                # Llama-2's explicit reshape(bsz, q_len, hidden_size)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, (attn_weights if output_attentions else None)

            return _forward

        # Replace the forward method on this specific attention instance
        attn.forward = _make_forward(attn, layer_idx)

    return model


def set_lahis_head_mask(model, head_mask):
    """
    Distribute a [num_layers, num_heads] head_mask tensor to every patched
    LlamaAttention layer so it is applied on the next forward pass.

    The slice head_mask[i] is a view of head_mask, so autograd gradients
    accumulate into head_mask.grad correctly (used by LAHIS importance scoring).
    """
    for i, layer in enumerate(model.model.layers):
        layer.self_attn._lahis_mask = head_mask[i]


def clear_lahis_head_mask(model):
    """Remove the head mask from all patched layers (restores normal behaviour)."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "_lahis_mask"):
            layer.self_attn._lahis_mask = None


# ---------------------------------------------------------------------------
# Load model & tokenizer
# ---------------------------------------------------------------------------

def load_model(
    model_name,
    device,
    half_precision=True,
    local=True,
    load_in_4bit=False,
    load_in_8bit=False,
):
    """
    Load a model and tokenizer, then apply the LAHIS head-mask patch.

    model_name options:
      "llama2"       → Llama-2-7B (requires HF_TOKEN, gated model)
      "olmo2"        → allenai/OLMo-2-1124-7B base  (open, no token needed)
      "hf:some/path" → any HuggingFace model path, e.g. "hf:jonny/olmo2-finetuned"
                       Use this for Jonny's finetuned model once he uploads it.
    """
    # ── Resolve model path ────────────────────────────────────────────────────
    # If model_name starts with "hf:" we treat the rest as a raw HuggingFace path.
    # This lets us load Jonny's finetuned model without adding a new constant.
    # Example: load_model("hf:jonny-username/olmo2-finetuned", device="mps", ...)
    if model_name.startswith("hf:"):
        model_path = model_name[3:]   # strip the "hf:" prefix
        patch_type = "olmo2"          # assume OLMo-2 architecture for finetuned models
    elif local:
        model_dict = {
            "aya":    AYA_NAME_LOCAL,
            "mistral": MISTRAL_NAME_LOCAL,
            "llama3": LLAMA3_NAME_LOCAL,
            "llama2": LLAMA2_NAME_LOCAL,
            "olmo2":  OLMO2_NAME_LOCAL,   # ← NEW
        }
        model_path = model_dict.get(model_name)
        patch_type = model_name
    else:
        model_dict = {
            "aya":    AYA_NAME,
            "mistral": MISTRAL_NAME,
            "llama3": LLAMA3_NAME,
            "llama2": LLAMA2_NAME,
            "olmo2":  OLMO2_NAME,         # ← NEW
        }
        model_path = model_dict.get(model_name)
        patch_type = model_name

    if not model_path:
        print("Wrong model name.")
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         "Use: llama2, olmo2, or hf:<huggingface/path>")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load model weights ────────────────────────────────────────────────────
    model_kwargs = {"token": HF_TOKEN}
    is_quantized = load_in_4bit or load_in_8bit

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one quantization mode: 4-bit or 8-bit.")

    if is_quantized:
        if BitsAndBytesConfig is None:
            raise ImportError(
                "BitsAndBytesConfig is unavailable. Install compatible "
                "transformers and bitsandbytes in Colab."
            )
        if "cuda" not in str(device):
            raise ValueError("4-bit/8-bit loading currently requires a CUDA device.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if half_precision else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
        print(f"Load model in {'4-bit' if load_in_4bit else '8-bit'} quantized mode")
    elif half_precision:
        model_kwargs["dtype"] = torch.bfloat16
        print("Load model in torch.bfloat16")

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.eval()
    if not is_quantized:
        model.to(device)

    # ── Apply LAHIS patch ─────────────────────────────────────────────────────
    # Each architecture needs its own patch because the attention forward()
    # method is slightly different. We pick the right one based on model type.
    if patch_type == "llama2":
        print("Applying LAHIS head-mask patch to Llama-2 attention layers...")
        patch_llama_for_head_mask(model)
        print("  Patch applied.")
    elif patch_type in ("olmo2", "hf"):
        # This covers both the base OLMo-2-7B AND Jonny's finetuned version —
        # finetuning with LoRA does not change the model architecture,
        # so the same patch works for base and finetuned.
        print("Applying LAHIS head-mask patch to OLMo-2 attention layers...")
        patch_olmo2_for_head_mask(model)
        print("  Patch applied.")

    # ── Report memory (CUDA only) ─────────────────────────────────────────────
    if torch.cuda.is_available() and "cuda" in str(device):
        memory_allocated_gb = torch.cuda.memory_allocated(device=device) / (1024 ** 3)
        print(f"** Memory allocated by model: {memory_allocated_gb:.2f} GB\n")

    return model, tokenizer
