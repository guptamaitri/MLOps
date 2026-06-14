import json

def test_columns_file_exists():
    with open(
        "models/columns.json",
        "r"
    ) as f:
        columns=json.load(f)
    assert "data_columns" in columns

def test_columns_not_empty():
    with open(
        "models/columns.json",
        "r"
    ) as f:
        columns=json.load(f)
    assert len(
        columns["data_columns"]
    ) > 0

def test_no_duplicate_columns():
    with open(
        "models/columns.json",
        "r"
    ) as f:
        columns=json.load(f)
    assert len(
        columns["data_columns"]

    ) == len(
        set(columns["data_columns"])
    )