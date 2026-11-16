"""Dense segment partitions must preserve every byte exactly once and in order."""
import pytest
from merlin.compile.application_deployment import _partition, PagedParameters
from merlin.semantic_compiler.target_binding import NativeCompilationError


def test_partition_accepts_tail_and_rejects_gaps_overlaps_and_reordering():
    _partition([dict(source_offset=0, bytes=8), dict(source_offset=8, bytes=3)], 11)
    for rows in ([dict(source_offset=0, bytes=8), dict(source_offset=9, bytes=2)],
                 [dict(source_offset=0, bytes=8), dict(source_offset=7, bytes=4)],
                 [dict(source_offset=8, bytes=3), dict(source_offset=0, bytes=8)],
                 [dict(source_offset=0, bytes=8)],
                 [dict(source_offset=0, bytes=0), dict(source_offset=0, bytes=11)]):
        with pytest.raises(NativeCompilationError, match='deployment_partition'):
            _partition(rows, 11)
