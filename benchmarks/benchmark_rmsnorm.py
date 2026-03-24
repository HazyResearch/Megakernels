import torch
from .common import benchmark


@torch.library.custom_op("megakittens::rms_norm", mutates_args=())
def rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rms_norm(x, [x.shape[-1]], weight, eps)

@rms_norm_op.register_fake
def _rms_norm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.megakittens.rms_norm(x, weight, 1e-6)


def benchmark_rmsnorm() -> None:
    print("RMSNorm (bf16)")
    print(f"{chr(39)}shape{chr(39):>20}  {chr(39)}MK (us){chr(39):>10}  {chr(39)}PT (us){chr(39):>10}  {chr(39)}MK GB/s{chr(39):>10}  {chr(39)}PT GB/s{chr(39):>10}  {chr(39)}ratio{chr(39):>7}")
    print("-" * 78)

    for M, N in [
        (32768, 2048),
        (32768, 4096),
        (32768, 8192),
        (32768, 16384),
    ]:
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, dtype=torch.bfloat16, device="cuda")

        mk_ms, pt_ms = benchmark(rms_norm, (x, w))

        bytes_moved = M * N * 2 * 2 + N * 2
        mk_gbps = bytes_moved / mk_ms / 1e6
        pt_gbps = bytes_moved / pt_ms / 1e6

        print(f"  ({M:>5}, {N:>5})  {mk_ms*1000:>10.2f}  {pt_ms*1000:>10.2f}  {mk_gbps:>10.1f}  {pt_gbps:>10.1f}  {pt_ms/mk_ms:>6.2f}x")


if __name__ == "__main__":
    benchmark_rmsnorm()
