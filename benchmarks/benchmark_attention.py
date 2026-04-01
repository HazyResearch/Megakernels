import torch
from .common import benchmark


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.attention(q, k, v)


def benchmark_attention() -> None:
    print("Attention (bf16, head_dim=128)")
    print(f"{'shape':>28}  {'MK (us)':>10}  {'PT (us)':>10}  {'MK TF':>8}  {'PT TF':>8}  {'ratio':>7}")
    print("-" * 80)

    head_dim = 128
    for batch, seq_len, num_heads in [
        (16, 1024, 16),
        (16, 2048, 16),
        (16, 4096, 16),
        (16, 8192, 16),
        (16, 16384, 16),
    ]:
        q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(attention, (q, k, v))

        flops = 4.0 * batch * num_heads * seq_len * seq_len * head_dim
        mk_tf = flops / mk_ms / 1e9
        pt_tf = flops / pt_ms / 1e9

        print(f"  (b={batch:>2} s={seq_len:>5} h={num_heads:>2})  {mk_ms*1000:>10.1f}  {pt_ms*1000:>10.1f}  {mk_tf:>8.1f}  {pt_tf:>8.1f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_attention()
