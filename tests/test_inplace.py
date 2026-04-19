import torch

import megakittens
from .common import check


@torch.library.custom_op("megakittens::add_inplace", mutates_args=("a",))
def add_inplace_op(a: torch.Tensor, b: torch.Tensor) -> None:
    a.add_(b)


@add_inplace_op.register_fake
def _add_inplace_fake(a: torch.Tensor, b: torch.Tensor) -> None:
    pass


def _resolve_add_inplace(args: tuple, kwargs: dict) -> tuple["AddInplace", list[int]]:
    # auto_functionalized_v2 wraps this void-returning mutating op, producing a
    # (None, a_copy) tuple. Our single itype output maps to aten tuple index 1.
    return AddInplace(), [1]


class AddInplace(megakittens.schema.itype.IType):
    """Temporary Itype just for testing inplace ops"""
    TILE_SIZE = 128
    MAX_TILES_PER_INST = 2
    TMA = megakittens.jit.pykittens.st(
        dtype=megakittens.schema.dtype.DType.bf16, rows=TILE_SIZE, cols=TILE_SIZE,
    )

    torch_functions_map = {
        torch.ops.megakittens.add_inplace: _resolve_add_inplace,
        torch.ops.megakittens.add_inplace.default: _resolve_add_inplace,
    }

    test_cases = [((), (128, 128))]
    bench_cases = [((), (4096, 4096))]
    test_atol = 0.2
    test_rtol = 1e-2

    @staticmethod
    def test_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        torch.ops.megakittens.add_inplace(a, b)
        return a

    def test_args(self, case: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(*case, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(*case, dtype=torch.bfloat16, device="cuda")
        return (a, b)

    @property
    def cpp_include(self) -> str:
        return "itypes/elementwise_binary.cuh"

    @property
    def cpp_template(self) -> str:
        return "ElementwiseBinary<MKConfig, MKGlobals, BinaryOps<BinaryOp::ADD>, {tensors}>"

    @property
    def inputs(self) -> list[megakittens.schema.tensor.TensorSpec]:
        spec = megakittens.schema.tensor.TensorSpec(
            dtype=megakittens.schema.dtype.DType.bf16,
            granularity=(self.TILE_SIZE, self.TILE_SIZE),
            tma_types=[self.TMA],
        )
        return [spec, spec]

    @property
    def outputs(self) -> list[megakittens.schema.tensor.TensorSpec]:
        return [megakittens.schema.tensor.TensorSpec(
            dtype=megakittens.schema.dtype.DType.bf16,
            granularity=(self.TILE_SIZE, self.TILE_SIZE),
            tma_types=[self.TMA],
        )]

    @property
    def tiles_per_inst(self) -> int:
        return min(megakittens.dispatcher.Dispatcher.NUM_PAGES // 2, self.MAX_TILES_PER_INST)

    @property
    def inplace_mapping(self) -> dict[int, int]:
        return {0: 0}

    def block_indices(
        self,
        src_metas: tuple[megakittens.schema.tensor.TensorMeta, ...],
        dst_metas: tuple[megakittens.schema.tensor.TensorMeta, ...],
        src_ranges: tuple[megakittens.schema.tensor.TensorRange, ...],
        dst_ranges: tuple[megakittens.schema.tensor.TensorRange, ...],
    ) -> list[tuple[int, ...]]:
        dst_range = dst_ranges[0]
        indices: list[tuple[int, ...]] = []
        for b in range(dst_range[0].size):
            for d in range(dst_range[1].size):
                for r in range(dst_range[2].size // self.TILE_SIZE):
                    for c in range(0, dst_range[3].size // self.TILE_SIZE, self.tiles_per_inst):
                        n = min(self.tiles_per_inst, dst_range[3].size // self.TILE_SIZE - c)
                        index: list[int] = []
                        for src_range in src_ranges:
                            index.extend([
                                src_range[0].start + b,
                                src_range[1].start + d,
                                src_range[2].start // self.TILE_SIZE + r,
                                src_range[3].start // self.TILE_SIZE + c,
                            ])
                        index.extend([
                            dst_range[0].start + b,
                            dst_range[1].start + d,
                            dst_range[2].start // self.TILE_SIZE + r,
                            dst_range[3].start // self.TILE_SIZE + c,
                        ])
                        index.append(n)
                        indices.append(tuple(index))
        return indices

    def access_regions(
        self,
        block_index: tuple[int, ...],
        src_metas: tuple[megakittens.schema.tensor.TensorMeta, ...],
        dst_metas: tuple[megakittens.schema.tensor.TensorMeta, ...],
    ) -> tuple[list[list[tuple[tuple[int, int], ...]]], list[list[tuple[tuple[int, int], ...]]]]:
        n = block_index[-1]
        src_regions: list[list[tuple[tuple[int, int], ...]]] = []
        for i in range(len(self.inputs)):
            b, d, r, c = block_index[i * 4: i * 4 + 4]
            src_regions.append([(
                (b, b + 1), (d, d + 1),
                (r * self.TILE_SIZE, (r + 1) * self.TILE_SIZE),
                (c * self.TILE_SIZE, (c + n) * self.TILE_SIZE),
            )])
        b, d, r, c = block_index[len(self.inputs) * 4: len(self.inputs) * 4 + 4]
        dst_region = (
            (b, b + 1), (d, d + 1),
            (r * self.TILE_SIZE, (r + 1) * self.TILE_SIZE),
            (c * self.TILE_SIZE, (c + n) * self.TILE_SIZE),
        )
        return src_regions, [[dst_region]]

    def validate(
        self,
        src_metas: tuple[megakittens.schema.tensor.TensorMeta, ...],
        dst_metas: tuple[megakittens.schema.tensor.TensorMeta, ...],
        src_ranges: tuple[megakittens.schema.tensor.TensorRange, ...],
        dst_ranges: tuple[megakittens.schema.tensor.TensorRange, ...],
    ) -> None:
        super().validate(src_metas, dst_metas, src_ranges, dst_ranges)
        for i, src_range in enumerate(src_ranges):
            if src_range.effective_shape != dst_ranges[0].effective_shape:
                raise RuntimeError(
                    f"[MegaKittens] AddInplace effective shape mismatch at src {i}: "
                    f"src={src_range.effective_shape} dst={dst_ranges[0].effective_shape}"
                )


def test_add_inplace() -> None:
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        torch.ops.megakittens.add_inplace(a, b)
        return a

    for shape in [(128, 128), (512, 1024), (1280, 2048), (2, 128, 256), (3, 512, 1024), (2, 3, 128, 256)]:
        a = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        check(f, (a, b), atol=0.2, rtol=1e-2)


def test_add_inplace_sibling_pre_mutation_reader() -> None:
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        q = torch.neg(a)
        torch.ops.megakittens.add_inplace(a, b)
        return q + a

    a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    check(f, (a, b), atol=0.5, rtol=1e-2)


def test_add_inplace_chained() -> None:
    def f(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        torch.ops.megakittens.add_inplace(a, b)
        torch.ops.megakittens.add_inplace(a, c)
        return a

    a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    c = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    check(f, (a, b, c), atol=0.5, rtol=1e-2)


def test_add_inplace_followed_by_further_compute() -> None:
    def f(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        torch.ops.megakittens.add_inplace(a, b)
        return a + c

    a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    c = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    check(f, (a, b, c), atol=0.5, rtol=1e-2)


def test_add_inplace_on_intermediate_with_sibling_reader() -> None:
    def f(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.relu(a)
        q = torch.neg(x)
        torch.ops.megakittens.add_inplace(x, b)
        return q + x

    a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    check(f, (a, b), atol=0.5, rtol=1e-2)


if __name__ == "__main__":
    test_add_inplace()
    test_add_inplace_sibling_pre_mutation_reader()
    test_add_inplace_chained()
    test_add_inplace_followed_by_further_compute()
    test_add_inplace_on_intermediate_with_sibling_reader()
