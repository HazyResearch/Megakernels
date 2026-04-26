# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Union

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

# Custom inductor options
torch._inductor.config.aggressive_fusion = True
torch._inductor.config.combo_kernels = True
torch._inductor.config.benchmark_fusion = True  # defaults to False
torch._inductor.config.triton.coalesce_tiling_analysis = True

# PDL must be set before torch.compile is called, so check sys.argv early.
if hasattr(torch._inductor.config.triton, "enable_pdl"):
    import re
    from torch._inductor.codecache import PyCodeCache
    torch._inductor.config.triton.enable_pdl = "--pdl" in sys.argv
    def set_inductor_pdl(enabled: bool):
        updated = 0
        for mod in getattr(PyCodeCache, "modules", []):
            if not hasattr(mod, "call"):
                continue
            src_file = getattr(mod, "__file__", None)
            if src_file is None:
                continue
            try:
                src = Path(src_file).read_text()
            except OSError:
                continue

            # Strip any existing launch_pdl= so we can re-insert cleanly
            src = re.sub(r",\s*launch_pdl\s*=\s*\w+", "", src)

            if enabled:
                patched_src = re.sub(
                    r",(\s*stream\s*=\s*\w+\s*\))",
                    r", launch_pdl=True,\1",
                    src,
                )
            else:
                patched_src = src

            exec(compile(patched_src, src_file, "exec"), mod.__dict__)  # noqa: S102
            updated += 1

        total = len(getattr(PyCodeCache, "modules", []))
        label = "True" if enabled else "False"
        if updated == 0:
            print(f"[patch_pdl] WARNING: no inductor modules with call() found (total in cache: {total})")
        else:
            print(f"[patch_pdl] Set launch_pdl={label} on {updated}/{total} inductor module(s)")


COMPILE_MODES = {"none", "default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune", "megakittens"}
COMPILE_ARGS = {
    "none": None,
    "default": {"fullgraph": True},
    "reduce-overhead": {"fullgraph": True, "mode": "reduce-overhead"},
    "max-autotune-no-cudagraphs": {"fullgraph": True, "mode": "max-autotune-no-cudagraphs"},
    "max-autotune": {"fullgraph": True, "mode": "max-autotune"},
    "megakittens": None,
}

create_block_mask = torch.compile(create_block_mask, options={"combo_kernels": False})

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    # input_pos: [B, S]
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(dtype=torch.int)

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask) -> torch.Tensor:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(dtype=torch.int)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int):
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    new_tokens = []
    for _ in range(num_new_tokens):
        next_token = decode_one_token(model, cur_token, input_pos, block_mask)
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.clone()

    return new_tokens

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(batch_size, -1), input_pos).clone()
    seq[:, T] = next_token.squeeze()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1)
    seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    metrics = {'decode_time': t1 - t0}

    return seq, metrics

def encode_tokens(tokenizer, string, bos=True):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device='cuda')

def _load_model(checkpoint_path, precision, use_tp):
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device='cuda', dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

def main(
    prompt: Union[int, str] = "Hello, my name is",
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    warmup: int = 5,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: str = "none",
    compile_prefill: bool = False,
    pdl: bool = False,
    model_name: str = None,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    precision = torch.bfloat16

    if model_name is not None:
        assert isinstance(prompt, int), \
            "--prompt must be an integer (token count) when using --model_name"
        print(f"Creating synthetic {model_name} model ...")
        t0 = time.time()
        model = Transformer.from_name(model_name)
        for p in model.parameters():
            p.data = torch.randn_like(p, device='cuda', dtype=precision)
        model = model.to(device='cuda', dtype=precision).eval()
        tokenizer = None
        encoded = torch.randint(0, 1024, (prompt,), device='cuda', dtype=torch.int64)
    else:
        assert checkpoint_path.is_file(), checkpoint_path
        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), str(tokenizer_path)
        print("Loading model ...")
        t0 = time.time()
        model = _load_model(checkpoint_path, precision, use_tp)
        tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
        if isinstance(prompt, str):
            encoded = encode_tokens(tokenizer, prompt, bos=True)
        else:
            # generate a fully synthetic prompt
            encoded = torch.randint(0, 1024, (prompt,), device='cuda', dtype=torch.int64)

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)

    if compile == "megakittens":
        import megakittens
        model = megakittens.compile(model, dry_run=True, save_dag=True)
    else:
        compile_opts = COMPILE_ARGS[compile]
        if compile_opts is not None:
            global decode_one_token, prefill
            decode_one_token = torch.compile(decode_one_token, **compile_opts)
            if compile_prefill:
                prefill = torch.compile(prefill, dynamic=True, **compile_opts)

    aggregate_metrics = {
        'tokens_per_sec': [],
    }
    start = -warmup if warmup > 0 else 0

    for i in range(start, num_samples):
        y, metrics = generate(
            model,
            encoded,
            max_new_tokens,
            batch_size=batch_size,
        )
        if i < 0:
            if i == start and compile_opts is not None:
                # Patch PDL after first run so compiled modules exist to patch.
                # Remaining warmup runs will capture CUDA graphs with PDL active.
                set_inductor_pdl(pdl)
            continue
        t = metrics['decode_time']

        if tokenizer is not None:
            if batch_size > 1:
                print("Only displaying the first generation of the batch")
            print(tokenizer.decode(y[0].tolist()))
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        total_tokens_sec = y.numel() / t
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print()
    print("==========")
    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default="Hello, my name is", help="Input prompt. If it's an integer, will instead generate a synthetic prompt.")
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup runs.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', default='none', choices=COMPILE_MODES, help='Compile mode.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--pdl', action='store_true', help='Enable Triton PDL (programmatic dependent launch).')
    parser.add_argument('--model_name', type=str, default=None, help='Use random weights for this model (e.g. llama-3.3-70b). Skips download.')

    args = parser.parse_args()
    main(
        args.prompt, args.num_samples, args.max_new_tokens, args.batch_size, args.warmup,
        args.checkpoint_path, args.compile, args.compile_prefill, args.pdl, args.model_name
    )
