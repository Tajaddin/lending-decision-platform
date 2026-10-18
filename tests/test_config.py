from __future__ import annotations

import pytest
from pydantic import SecretStr

from ldp.config import Settings


def test_client_keys_parse_pairs() -> None:
    s = Settings(api_keys=SecretStr("branch-app:aaaaaaaaaaaaaaaa1, batch:bbbbbbbbbbbbbbbb2"))
    assert s.client_keys() == {"aaaaaaaaaaaaaaaa1": "branch-app", "bbbbbbbbbbbbbbbb2": "batch"}


@pytest.mark.parametrize("raw", ["nocolon", "client:short", ":aaaaaaaaaaaaaaaaaaaa"])
def test_malformed_keys_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="client:key"):
        Settings(api_keys=SecretStr(raw)).client_keys()


def test_lake_paths() -> None:
    s = Settings(lake_root="/Volumes/workspace/lending/lake/")
    assert s.lake("silver") == "/Volumes/workspace/lending/lake/silver"
