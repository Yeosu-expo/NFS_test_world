from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from huggingface_hub import login
import torch
import os
import deepspeed

def get_tokenizer(model_type):
    model_name = "meta-llama/Llama-3.2-1B"
    opt_model = "facebook/opt-125m"

    if model_type == "opt":
        tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
    elif model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model(model_type, tokenizer, device, dtype=torch.float16, pretrain=False):
    llama_1b = "meta-llama/Llama-3.2-1B"
    opt_125m = "facebook/opt-125m"

    if model_type == "opt":
        if pretrain:
            # from_config에는 dtype 인자를 넣지 않습니다. 생성 후 .to(dtype)로 지정합니다.
            cfg = AutoConfig.from_pretrained(opt_125m)
            model = AutoModelForCausalLM.from_config(cfg)
        else:
            # from_pretrained는 torch_dtype 키워드를 사용합니다.
            model = AutoModelForCausalLM.from_pretrained(opt_125m, torch_dtype=dtype)
    elif model_type == "llama":
        if pretrain:
            cfg = LlamaConfig.from_pretrained(llama_1b)
            model = LlamaForCausalLM(cfg)
        else:
            model = LlamaForCausalLM.from_pretrained(llama_1b, torch_dtype=dtype)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 디바이스/데이터타입 지정
    model = model.to(device)
    if pretrain:
        model = model.to(dtype=dtype)

    # ensure embedding size matches tokenizer size (important when training from scratch)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    return model.train()

def login_hf(hug_token: str | None):
    """Login only if a real token (starts with 'hf_') is available."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or hug_token
    if not token:
        return
    if isinstance(token, str) and token.startswith("hf_"):
        try:
            login(token=token, add_to_git_credential=False)
        except Exception:
            # Don't crash multi-node run because of a login hiccup
            pass
    # else: token name like 'token240108' -> skip silently

def get_model_engine(model, ds_config_path):
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path
    )

    return model_engine, optimizer