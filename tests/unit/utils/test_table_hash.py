"""hash_pyarrow_table depends on content only."""

import pyarrow as pa
import pytest

from modularml.utils.data.pyarrow_data import hash_pyarrow_table


@pytest.mark.unit
def test_table_hash_ignores_chunking_and_metadata_but_not_values():
    a = pa.table({"x": [1.0, 2.0, 3.0], "s": ["a", "b", "c"], "v": [[1, 2], [3], []]})
    chunked = pa.concat_tables([a.slice(0, 1), a.slice(1)])
    with_meta = a.replace_schema_metadata({"k": "v"})
    via_pandas = pa.Table.from_pandas(a.to_pandas())
    h = hash_pyarrow_table(a)
    assert hash_pyarrow_table(chunked) == h
    assert hash_pyarrow_table(with_meta) == h
    assert hash_pyarrow_table(via_pandas) == h

    changed = pa.table(
        {"x": [1.0, 2.0, 3.5], "s": ["a", "b", "c"], "v": [[1, 2], [3], []]},
    )
    assert hash_pyarrow_table(changed) != h
    renamed = a.rename_columns(["x", "s", "w"])
    assert hash_pyarrow_table(renamed) != h
