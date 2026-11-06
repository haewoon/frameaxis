from frameaxis.semaxis import CoreUtil
import pytest


def test_core_load_conceptnet_axes():
    assert len(CoreUtil.load_conceptnet_antonyms_axes()) == 732


def test_core_load_plutchik_wheel_of_emotions_axes():
    assert len(CoreUtil.load_plutchik_wheel_of_emotions_axes()) == 24


def test_load_wordnet_antonyms_axes():
    assert len(CoreUtil.load_wordnet_antonyms_axes()) == 1828


@pytest.mark.parametrize("baseline, moment2, moment4", [
    (None, 0.04, 0.0016),
    (0.0, 0.2, 0.0656),
    (0.1, 0.13, 0.0313),
])
def test_moments_preserve_supplied_baseline(baseline, moment2, moment4):
    data = {0.2: 1, 0.6: 1}
    assert CoreUtil._moment_dict(data, 2, baseline) == pytest.approx(moment2)
    assert CoreUtil._moment_dict(data, 4, baseline) == pytest.approx(moment4)


def test_kurtosis_preserves_zero_baseline():
    axis = ("negative", "positive")
    result = CoreUtil.compute_kurtosis({axis: {0.2: 1, 0.6: 1}}, {axis: 0.0})
    assert result[0][:2] == list(axis)
    assert result[0][2] == pytest.approx(-1.36)
