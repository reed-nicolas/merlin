"""Compose independent output tiles with ordered reduction tiles.

The target supplies the tile geometry, numerical policy, and separately
qualified pad bytes. Each resulting request retains the source's ordered K
sequence; this module only moves declared boundary bytes and output tiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .model import KernelRequest
from .ordered_k_tiling import (
    OrderedKTiling,
    materialize_row_major_k_slices,
    partition_ordered_matmul_k,
)
from .ordered_mn_tiling import (
    OrderedMNTiling,
    assemble_row_major_mn_outputs,
    materialize_row_major_mn_inputs,
    partition_ordered_matmul_mn,
)


class OrderedMNKTilingError(ValueError):
    """An outer tile plan no longer describes the selected source exactly."""


@dataclass(frozen=True)
class OrderedMNKTiling:
    source: KernelRequest
    mn: OrderedMNTiling
    k_tiles: tuple[OrderedKTiling, ...]
    tile_k: int
    qualified_policy: str
    k_head_pad_element: bytes | None = None

    def __post_init__(self) -> None:
        if self.mn.source != self.source or len(self.k_tiles) != len(self.mn.tiles):
            raise OrderedMNKTilingError("M/N/K tiles do not cover the selected source")
        for output_tile, reduction in zip(self.mn.tiles, self.k_tiles):
            expected = partition_ordered_matmul_k(
                output_tile.request,
                tile_k=self.tile_k,
                qualified_policy=self.qualified_policy,
                head_pad_element=self.k_head_pad_element,
            )
            if reduction != expected:
                raise OrderedMNKTilingError("K tile differs from its output region")

    @property
    def requests(self) -> tuple[KernelRequest, ...]:
        """One separately compilable, ordered-reduction request per output tile."""
        return tuple(reduction.request for reduction in self.k_tiles)

    def record(self) -> dict[str, object]:
        return {
            "schema": "merlin.ordered_mnk_tiling.v1",
            "source_request_digest": self.source.digest(),
            "mn": self.mn.record(),
            "tile_k": self.tile_k,
            "qualified_policy": self.qualified_policy,
            "k_head_pad_element_hex": (
                None if self.k_head_pad_element is None else self.k_head_pad_element.hex()
            ),
            "k": [reduction.record() for reduction in self.k_tiles],
        }


def partition_ordered_matmul_mnk(
    source: KernelRequest, *, tile_m: int, tile_n: int, tile_k: int,
    qualified_policy: str, mn_tail_pad_element: bytes | None = None,
    k_head_pad_element: bytes | None = None,
) -> OrderedMNKTiling:
    """Partition one oriented contraction without changing valid-cell K order.

    The first short K tile is head padded, so real source terms still appear
    in increasing K order. The caller must qualify that pad as an exact reset
    identity. M/N padding is a separate caller premise for invalid output
    lanes, and output assembly drops those lanes.
    """
    mn = partition_ordered_matmul_mn(
        source, tile_m=tile_m, tile_n=tile_n,
        qualified_policy=qualified_policy, tail_pad_element=mn_tail_pad_element,
    )
    reductions = tuple(
        partition_ordered_matmul_k(
            output_tile.request, tile_k=tile_k,
            qualified_policy=qualified_policy, head_pad_element=k_head_pad_element,
        )
        for output_tile in mn.tiles
    )
    return OrderedMNKTiling(source, mn, reductions, tile_k, qualified_policy,
                            k_head_pad_element)


def materialize_row_major_mnk_inputs(
    tiling: OrderedMNKTiling, source_bytes: Mapping[str, bytes], *, element_bytes: int,
) -> tuple[dict[str, bytes], ...]:
    """Pack each output tile's complete, ordered K boundary from source bytes."""
    mn_inputs = materialize_row_major_mn_inputs(
        tiling.mn, source_bytes, element_bytes=element_bytes,
    )
    return tuple(
        materialize_row_major_k_slices(reduction, inputs, element_bytes=element_bytes)
        for reduction, inputs in zip(tiling.k_tiles, mn_inputs)
    )


def assemble_row_major_mnk_outputs(
    tiling: OrderedMNKTiling, tile_outputs: Sequence[bytes], *, element_bytes: int,
) -> bytes:
    """Trim invalid M/N lanes and join final accumulated output tiles."""
    return assemble_row_major_mn_outputs(
        tiling.mn, tile_outputs, element_bytes=element_bytes,
    )
