import json

import pytest

from exoplanet.data.catalogue import (
    build_query,
    catalogue_path,
    fetch_koi_catalogue,
    load_koi_catalogue,
    metadata_path,
)

CSV = (
    "kepid,kepoi_name,koi_disposition,koi_period\n"
    "1,K00001.01,CONFIRMED,2.47\n"
    "1,K00001.02,FALSE POSITIVE,10.1\n"
    "2,K00002.01,CANDIDATE,5.0\n"
)


class FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.content = text.encode()
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self.text)


@pytest.fixture
def config(tmp_path):
    return {
        "paths": {"raw_dir": tmp_path / "raw"},
        "catalogue": {
            "tap_url": "https://example.invalid/TAP/sync",
            "table": "cumulative",
            "filename": "koi.csv",
            "timeout_s": 5,
            "retries": 0,
        },
    }


def test_build_query():
    assert build_query("cumulative") == "select * from cumulative"
    assert build_query("cumulative", ["kepid", "koi_period"]) == (
        "select kepid, koi_period from cumulative"
    )


def test_fetch_writes_csv_and_metadata(config):
    session = FakeSession(CSV)
    path = fetch_koi_catalogue(config, session=session)

    assert path == catalogue_path(config)
    assert session.calls[0][1] == {"query": "select * from cumulative", "format": "csv"}
    meta = json.loads(metadata_path(path).read_text())
    assert meta["n_rows"] == 3
    assert meta["query"] == "select * from cumulative"
    assert len(meta["sha256"]) == 64
    assert len(load_koi_catalogue(config)) == 3


def test_fetch_uses_cache_unless_forced(config):
    fetch_koi_catalogue(config, session=FakeSession(CSV))
    session = FakeSession(CSV)
    fetch_koi_catalogue(config, session=session)
    assert session.calls == []
    fetch_koi_catalogue(config, force=True, session=session)
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        '<?xml version="1.0"?><VOTABLE><INFO name="QUERY_STATUS" value="ERROR"/></VOTABLE>',
        "kepid,koi_period\n1,2.0\n",
        "kepid,kepoi_name,koi_disposition\n",
        "kepid,kepoi_name,koi_disposition\n1,K1.01,CONFIRMED\n1,K1.01,CONFIRMED\n",
    ],
    ids=["votable-error", "missing-columns", "empty", "duplicate-koi"],
)
def test_fetch_rejects_invalid_responses_without_writing(config, text):
    with pytest.raises(ValueError):
        fetch_koi_catalogue(config, session=FakeSession(text))
    assert not catalogue_path(config).exists()


def test_load_without_cache_raises(config):
    with pytest.raises(FileNotFoundError):
        load_koi_catalogue(config)
