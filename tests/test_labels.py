import pandas as pd
import pytest

from exoplanet.config import load_config
from exoplanet.data.labels import LABEL, label_catalogue


@pytest.fixture
def catalogue():
    return pd.DataFrame(
        {
            "kepid": [1, 1, 2, 3, 4],
            "koi_disposition": [
                "CONFIRMED",
                "FALSE POSITIVE",
                "CANDIDATE",
                "FALSE POSITIVE",
                "CONFIRMED",
            ],
            "koi_parm_prov": [
                "q1_q17_dr25_koi",
                "q1_q17_dr25_koi",
                "q1_q17_dr25_koi",
                "q1_q16_koi",
                "q1_q17_dr25_koi",
            ],
        }
    )


@pytest.fixture
def config():
    return load_config()


def test_label_mapping_and_provenance_filter(catalogue, config):
    out = label_catalogue(catalogue, config)
    assert list(out["kepid"]) == [1, 1, 4]
    assert list(out[LABEL]) == [1, 0, 1]
    assert out[LABEL].dtype.kind == "i"
    assert "CANDIDATE" not in set(out["koi_disposition"])


def test_without_provenance_filter_keeps_all_labelled(catalogue, config):
    out = label_catalogue(catalogue, {**config, "sample": {"parameter_provenance": None}})
    assert list(out["kepid"]) == [1, 1, 3, 4]
    assert list(out[LABEL]) == [1, 0, 0, 1]


def test_input_is_not_modified(catalogue, config):
    before = catalogue.copy()
    label_catalogue(catalogue, config)
    pd.testing.assert_frame_equal(catalogue, before)


@pytest.mark.parametrize("bad", ["NOT DISPOSITIONED", None])
def test_unmapped_dispositions_raise(catalogue, config, bad):
    catalogue.loc[0, "koi_disposition"] = bad
    with pytest.raises(ValueError, match="Unmapped dispositions"):
        label_catalogue(catalogue, config)
