import numpy as np

from frameaxis.core import fdr_adjust, random_subset_null, two_sided_pvalue


def test_two_sided_catches_shift_toward_zero():
    # Cases from #13: the old sign rule gave p ~ 1 for both.
    rng = np.random.default_rng(1)
    null = rng.normal(0.05, 0.01, 1000)
    assert two_sided_pvalue(0.02, null) < 0.01
    null = rng.normal(-0.05, 0.01, 1000)
    assert two_sided_pvalue(-0.01, null) < 0.01


def test_two_sided_is_symmetric_and_never_zero():
    null = np.linspace(-1, 1, 999)
    assert np.isclose(two_sided_pvalue(0.9, null), two_sided_pvalue(-0.9, null))
    assert two_sided_pvalue(100.0, null) == 1 / 1000
    assert two_sided_pvalue(null.mean(), null) == 1.0


def test_two_sided_vectorized_over_axes():
    null = np.random.default_rng(2).normal(size=(500, 3))
    obs = np.array([0.0, 3.0, -3.0])
    p = two_sided_pvalue(obs, null)
    assert p.shape == (3,)
    for j in range(3):
        assert np.isclose(p[j], two_sided_pvalue(obs[j], null[:, j]))


def test_fdr_bh_matches_hand_computation():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # sorted p 0.01, 0.03, 0.04, 0.20 -> p*m/rank 0.04, 0.06, 0.0533, 0.20 -> monotone 0.04, 0.0533, 0.0533, 0.20
    np.testing.assert_allclose(fdr_adjust(p), [0.04, 0.16 / 3, 0.16 / 3, 0.20])
    assert np.all(fdr_adjust(p, "by") >= fdr_adjust(p))


def test_random_subset_null_is_seeded():
    values = np.arange(60, dtype=float).reshape(20, 3)
    a = random_subset_null(values, 5, 10, seed=7)
    b = random_subset_null(values, 5, 10, seed=7)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, random_subset_null(values, 5, 10, seed=8))


def test_calibration_under_the_null():
    # A random "subset" of the table should be flagged at about the nominal
    # rate uncorrected, and almost never after BH (the check suggested in #13).
    rng = np.random.default_rng(3)
    table = rng.normal(0.1, 1.0, size=(2000, 200))
    flagged, flagged_bh = [], []
    for trial in range(5):
        idx = rng.choice(table.shape[0], 150, replace=False)
        observed = table[idx].mean(axis=0)
        null = random_subset_null(table, 150, 999, seed=trial)
        p = two_sided_pvalue(observed, null)
        flagged.append(np.mean(p <= 0.05))
        flagged_bh.append(np.sum(fdr_adjust(p) <= 0.05))
    assert 0.02 < np.mean(flagged) < 0.09
    assert np.mean(flagged_bh) <= 1


def test_detects_a_real_shift_after_bh():
    rng = np.random.default_rng(4)
    table = rng.normal(0.0, 1.0, size=(2000, 50))
    idx = rng.choice(table.shape[0], 150, replace=False)
    observed = table[idx].mean(axis=0)
    observed[:5] += 0.6  # about 6 standard errors
    null = random_subset_null(table, 150, 9999, seed=0)
    q = fdr_adjust(two_sided_pvalue(observed, null))
    assert np.all(q[:5] <= 0.05)
