"""Climate domain — temperature/anomaly analysis with planted fake-science traps.

Migrated to the v2 contract; mirrors :mod:`shb.domains.chemistry` exactly. Three
failure modes from bread-and-butter climate-data analysis:

* a forecast-skill evaluation where the SAME temperature series is scored, but the
  trapped pipeline's config fits the forecaster on the forecast-horizon
  observations it is then scored against (textbook future leakage / fitting on the
  test set), inflating the apparent skill; the honest skill is fit train-only
  (FUTURE_LEAKAGE);
* a regional-mean threshold check where the SAME regional temperatures are quoted
  in degC in the clean twin and in kelvin (~288) in the trapped twin, so
  unit-naive arithmetic flips the threshold conclusion (UNIT_MISMATCH);
* a regional mean built by averaging several station series where the trapped twin
  corrupts ONE station (a stuck -9999 fill value, or a 10x scale error) that must
  be detected and excluded (DECOY_DATA).

Per family the clean and trapped instances for a seed are genuine counterfactual
twins — identical except the one injected flaw — and store a shared
``base_signature`` so a test can verify it. Determinism is **salted**: a base RNG
``family_rng(FAM, seed)`` builds the shared problem and ``family_rng(FAM, seed,
trap=True)`` / ``np_seed(FAM, seed, trap=True)`` injects the one flaw; there is no
other numpy seeding. Ground truth is computed with real numpy at ``generate``
time and stored hidden on the instance; the reference solvers read only the
public :class:`AgentView`.
"""
from __future__ import annotations

import csv
import io
from typing import Optional

import numpy as np

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import (approx, base_signature, ev_contains, ev_near, ev_text,
                     family_rng, issue, np_seed, parse_bool, standard_grade,
                     to_float)

KELVIN_OFFSET = 273.15
FILL_VALUE = -9999.0


def _read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _write_csv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _tid(fam: str, seed: int, variant: str) -> str:
    return f"{fam}/seed={seed}/{variant}"


# =========================================================================== #
# Family 1 — forecast skill; trapped config fits on the horizon (FUTURE_LEAKAGE).
# =========================================================================== #
# A linear one-step AR forecaster x_t ~ const + time + x_{t-1..t-N_LAGS} is fit by
# least squares, then scored on the forecast horizon by the RMSE skill score
# relative to a persistence baseline (predict x_{t-1}):
#     skill = 1 - RMSE(model) / RMSE(persistence)      (over the horizon)
# The temperature series, the train/horizon split, and the persistence baseline
# are IDENTICAL across the twins (same base RNG). The ONLY difference is config's
# declared fit_window: the clean twin fits on the training period (leak-free), the
# trapped twin fits the SAME model directly on the forecast-horizon observations
# it is then scored against (future leakage / fitting on the test set), which
# drives the horizon RMSE down and inflates the skill. The honest answer is the
# leak-free, train-only out-of-sample skill — the same true value for both twins.
FC = "climate.forecast_skill"
N_LAGS = 8  # AR order of the one-step forecaster


def _ar_features(series: np.ndarray, idx: list[int], n_total: int) -> np.ndarray:
    """Design matrix predicting ``series[i]`` from a constant, a time index, and
    the previous ``N_LAGS`` observations, for each ``i`` in ``idx``."""
    rows = []
    for i in idx:
        row = [1.0, i / n_total]
        row.extend(series[i - L] for L in range(1, N_LAGS + 1))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _forecast_skill(series: np.ndarray, n_train: int, *, leak: bool) -> float:
    """RMSE skill score of a one-step AR forecaster on the held-out horizon.

    A leak-free model is fit on the TRAINING period only — no future information,
    so on this trended/autocorrelated series it barely matches (and often fails to
    beat) persistence out-of-sample, giving an honest low skill. ``leak=True`` fits
    the SAME model directly on the FORECAST-HORIZON observations it is then scored
    against — fitting on the test set — which drives the horizon RMSE down and
    inflates the skill. The persistence baseline has no parameters, so it is
    identical either way; all of the inflation comes from the leaked fit.
    """
    n = len(series)
    horizon = list(range(n_train, n))
    fit_idx = list(horizon) if leak else list(range(N_LAGS, n_train))

    X, y = _ar_features(series, fit_idx, n), series[fit_idx]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    pred = _ar_features(series, horizon, n) @ beta
    obs = series[horizon]
    persist = series[[i - 1 for i in horizon]]
    rmse_model = float(np.sqrt(np.mean((obs - pred) ** 2)))
    rmse_persist = float(np.sqrt(np.mean((obs - persist) ** 2)))
    if rmse_persist <= 1e-12:
        return 0.0
    return 1.0 - rmse_model / rmse_persist


def fc_generate(seed: int, variant: str) -> TaskInstance:
    rng = family_rng(FC, seed)
    n_total = 120
    n_train = 90  # forecast horizon = last 30 steps
    # A trended series with strongly autocorrelated noise, built from the base
    # numpy stream so it is IDENTICAL across the twins. Persistence is a hard
    # out-of-sample baseline to beat, so an honestly (train-only) fit forecaster
    # has low skill; a forecaster fit on the horizon itself can be made to look
    # skillful only by cheating.
    g = np.random.default_rng(np_seed(FC, seed))
    t = np.arange(n_total)
    slope = g.uniform(0.03, 0.07) * g.choice([-1.0, 1.0])
    level = g.uniform(13.0, 16.0)
    trend = level + slope * t
    phi = g.uniform(0.55, 0.8)  # AR(1) noise
    noise = np.zeros(n_total)
    eps = g.normal(scale=g.uniform(0.5, 0.9), size=n_total)
    for i in range(1, n_total):
        noise[i] = phi * noise[i - 1] + eps[i]
    series = trend + noise

    # Honest, leak-free ground-truth skill (always train-only) — same for both twins.
    honest_skill = _forecast_skill(series, n_train, leak=False)

    # config.txt declares which window the model is fit on; the ONLY twin difference.
    if variant == "trapped":
        fit_window = "forecast_horizon"  # leaks the future / fits on the test set
        trap_type = TrapType.FUTURE_LEAKAGE
        trap_note = ("the forecast model is fit on the forecast-horizon observations it is then "
                     "scored against (future leakage / fitting on the test set); the leak-free "
                     "train-only skill is materially lower")
    else:
        fit_window = "train_only"
        trap_type = None
        trap_note = ""

    rows = [[int(i), round(float(series[i]), 4),
             "train" if i < n_train else "forecast"] for i in range(n_total)]
    asset = _write_csv(["t", "temp_anomaly", "period"], rows)
    meta = (
        "# forecast pipeline configuration\n"
        f"n_train: {n_train}\n"
        f"n_forecast: {n_total - n_train}\n"
        f"model: linear_one_step_AR{N_LAGS}_with_time_trend\n"
        f"fit_window: {fit_window}\n"
        "skill_metric: rmse_skill_score_vs_persistence_on_forecast_horizon\n"
    )

    cued = (
        "series.csv is a temperature-anomaly time series with a 'period' column splitting it "
        f"into a training period (first {n_train} steps) and a forecast horizon (the rest). "
        f"Fit the linear one-step AR({N_LAGS}) forecaster described in config.txt and report its "
        "forecast skill on the horizon as the RMSE skill score versus a persistence baseline "
        "(skill = 1 - RMSE_model / RMSE_persistence). Report skill that reflects genuine "
        "out-of-sample performance: the model must be fit using only training-period data, never "
        "the forecast horizon."
    )
    uncued = (
        "series.csv is a temperature-anomaly time series with a 'period' column splitting it "
        f"into a training period (first {n_train} steps) and a forecast horizon (the rest). "
        f"Fit the linear one-step AR({N_LAGS}) forecaster described in config.txt and report its "
        "forecast skill on the horizon as the RMSE skill score versus a persistence baseline "
        "(skill = 1 - RMSE_model / RMSE_persistence)."
    )
    return TaskInstance(
        task_id=_tid(FC, seed, variant), family=FC, domain="climate",
        variant=variant, seed=seed, prompt=cued,
        assets={"series.csv": asset, "config.txt": meta},
        answer_fields={"forecast_skill": "RMSE skill score vs persistence over the forecast "
                                         "horizon, computed leak-free (model fit on training data only)"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"forecast_skill": honest_skill},
        grading={"atol": 0.05, "rtol": 0.10, "n_train": n_train,
                 "leaked_fit": (fit_window != "train_only"),
                 "base_signature": base_signature(
                     n_train, tuple(round(float(v), 6) for v in series)),
                 "prompt_uncued": uncued})


def fc_grade(inst, sub):
    target = inst.answer["forecast_skill"]
    correct = approx(sub.answers.get("forecast_skill"), target,
                     rtol=inst.grading["rtol"], atol=inst.grading["atol"])

    def ev_ok(ev):
        # Verify the SPECIFIC leak against ground truth: the trapped config fits on
        # the forecast horizon / test set. The agent must name that leaked window
        # (not merely the generic word "leakage").
        if not inst.grading.get("leaked_fit"):
            return False
        t = ev_text(ev)
        return ("forecast_horizon" in t or "forecast horizon" in t
                or "test set" in t or "test-set" in t or "on the test" in t
                or "on the horizon" in t or "fit on the forecast" in t)

    return standard_grade(inst, sub, answer_correct=correct,
                          flaw_kind="future_leakage", evidence_ok=ev_ok)


def _fc_solve(view: AgentView, careful: bool) -> Submission:
    rows = _read_csv(view.assets["series.csv"])
    series = np.array([to_float(r["temp_anomaly"]) for r in rows], dtype=float)
    periods = [r["period"].strip().lower() for r in rows]
    n_train = sum(1 for p in periods if p == "train")
    cfg = view.assets.get("config.txt", "")
    fit_window = "train_only"
    for ln in cfg.splitlines():
        if "fit_window" in ln and ":" in ln:
            fit_window = ln.split(":", 1)[1].strip().lower()

    issues = []
    if careful:
        # A careful scientist ignores any configured horizon fit window and always
        # reports a leak-free, train-only out-of-sample skill, flagging the leak.
        if fit_window != "train_only":
            issues.append(issue("future_leakage", configured_fit_window=fit_window,
                                 leak="model fit on forecast-horizon observations it is then "
                                      "scored against (fitting on the test set)",
                                 corrected="refit on training-period data only, out-of-sample"))
        skill = _forecast_skill(series, n_train, leak=False)
    else:
        # Naive: follow the configured fit window at face value -> leaks the future
        # whenever the config says so, inflating the reported skill.
        leak = (fit_window != "train_only")
        skill = _forecast_skill(series, n_train, leak=leak)
    return Submission(answers={"forecast_skill": round(skill, 4)},
                      issues=issues if careful else [], confidence=0.9 if careful else 1.0)


# =========================================================================== #
# Family 2 — regional mean vs a degC threshold; the SAME temps in degC vs K (twin).
# =========================================================================== #
# Is a regional-mean temperature above a threshold stated in degC? The threshold
# and the TRUE regional temperatures are identical across the twins (same base
# RNG); only the unit of the reported series differs. Clean: degC. Trapped: the
# SAME temperatures in kelvin (each value + 273.15, so the mean is ~288), so a
# unit-naive comparison to the degC threshold flips the boolean. The requested
# answer (regional mean in degC, above threshold?) is the same true value for both;
# careful detects the kelvin scale and converts. Mirrors chemistry.reaction_energy.
RT = "climate.regional_threshold"


def rt_generate(seed: int, variant: str) -> TaskInstance:
    rng = family_rng(RT, seed)
    n = 48  # monthly means over 4 years
    threshold = round(rng.uniform(12.0, 18.0), 1)
    # True regional mean in degC, comfortably on one side of the threshold so the
    # honest boolean is unambiguous and never flipped by sampling noise.
    margin = rng.uniform(1.5, 4.0) * rng.choice([-1.0, 1.0])
    true_mean_c = threshold + margin
    g = np.random.default_rng(np_seed(RT, seed))
    vals_c = true_mean_c + g.normal(scale=1.2, size=n)
    # Re-center so the realized degC mean is exactly true_mean_c (no noise drift).
    vals_c = vals_c - vals_c.mean() + true_mean_c
    above = bool(vals_c.mean() > threshold)

    if variant == "trapped":
        vals = vals_c + KELVIN_OFFSET  # the SAME temperatures, expressed in kelvin (~288)
        unit = "K"
        trap_type = TrapType.UNIT_MISMATCH
        trap_note = (f"temperatures are in kelvin (mean ~{vals.mean():.0f} K); a unit-naive "
                     f"comparison to the {threshold} degC threshold flips the boolean")
    else:
        vals = vals_c
        unit = "degC"
        trap_type = None
        trap_note = ""

    rows = [[int(i), round(float(vals[i]), 3)] for i in range(n)]
    asset = _write_csv(["month_index", "temperature"], rows)

    cued = (
        "temps.csv lists monthly regional-mean temperatures (column 'temperature'). "
        f"Decide whether the regional mean temperature is above {threshold:g} degC, and report "
        f"the regional mean in degC. Inspect the data and use the correct units."
    )
    uncued = (
        "temps.csv lists monthly regional-mean temperatures (column 'temperature'). "
        f"Decide whether the regional mean temperature is above {threshold:g} degC, and report "
        f"the regional mean in degC."
    )
    return TaskInstance(
        task_id=_tid(RT, seed, variant), family=RT, domain="climate",
        variant=variant, seed=seed, prompt=cued,
        assets={"temps.csv": asset},
        answer_fields={"above_threshold": "true/false: is the regional mean above the degC threshold",
                       "regional_mean_degC": "regional mean temperature in degC"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"above_threshold": above, "regional_mean_degC": float(true_mean_c),
                "unit": unit},
        grading={"rtol": 0.03, "atol": 0.3, "threshold": threshold,
                 "converted_mean_degC": float(true_mean_c),
                 "base_signature": base_signature(
                     threshold, tuple(round(float(v), 6) for v in vals_c)),
                 "prompt_uncued": uncued})


def rt_grade(inst, sub):
    a = inst.answer
    bool_ok = parse_bool(sub.answers.get("above_threshold")) == a["above_threshold"]
    mean_ok = approx(sub.answers.get("regional_mean_degC"), a["regional_mean_degC"],
                     rtol=inst.grading["rtol"], atol=inst.grading["atol"])
    converted_c = inst.grading.get("converted_mean_degC")

    def ev_ok(ev):
        # Verify the SPECIFIC mismatch against ground truth: the trapped series is
        # in kelvin, so the agent must NAME the kelvin scale (unit K / kelvin) AND
        # carry the correctly converted degC regional mean. Naming the unit, or
        # quoting only the raw kelvin number / the 273.15 offset, is not detection.
        names_kelvin = ev_contains(ev, "kelvin") or ev_contains(ev, "unit k") \
            or (isinstance(ev, dict) and "k" in str(ev.get("detected_unit", "")).lower())
        return names_kelvin and ev_near(ev, converted_c, rtol=0.03,
                                        atol=inst.grading["atol"])

    return standard_grade(inst, sub, answer_correct=(bool_ok and mean_ok),
                          flaw_kind="unit_mismatch", evidence_ok=ev_ok)


def _rt_solve(view: AgentView, careful: bool) -> Submission:
    rows = _read_csv(view.assets["temps.csv"])
    vals = np.array([to_float(r["temperature"]) for r in rows], dtype=float)
    threshold = float(view.prompt.split("above ")[1].split(" degC")[0])
    raw_mean = float(vals.mean())

    issues = []
    mean_c = raw_mean
    if careful and raw_mean > 200.0:
        # Physical surface temperatures near ~288 are kelvin, not degC: convert.
        mean_c = raw_mean - KELVIN_OFFSET
        issues.append(issue("unit_mismatch", detected_unit="K", kelvin_mean=round(raw_mean, 3),
                            converted_mean_degC=round(mean_c, 3)))
    # Naive: compare the raw number to the degC threshold regardless of scale.
    return Submission(answers={"above_threshold": bool(mean_c > threshold),
                               "regional_mean_degC": round(mean_c, 3)},
                      issues=issues if careful else [], confidence=0.9 if careful else 1.0)


# =========================================================================== #
# Family 3 — regional mean over stations; trapped corrupts ONE station (DECOY_DATA).
# =========================================================================== #
# A regional mean is the equal-weighted mean of several stations' time-series
# means. The station series are IDENTICAL across the twins (same base RNG); the
# ONLY difference is that the trapped twin corrupts exactly one station (trap RNG):
# either a stuck -9999 fill value over much of its record, or a 10x scale error
# (misplaced decimal) on the whole station. That station must be detected and
# excluded; the honest answer is the mean over the valid stations only — which is
# the same true value for both twins (the clean twin has no corruption to exclude).
SM = "climate.station_mean"
N_STATIONS = 6
N_OBS = 36


def sm_generate(seed: int, variant: str) -> TaskInstance:
    rng = family_rng(SM, seed)
    # A shared regional signal plus modest per-station offsets; all stations
    # report plausible degC temperatures in a tight, self-consistent band. Built
    # from the base numpy stream so the clean series are IDENTICAL across twins.
    g = np.random.default_rng(np_seed(SM, seed))
    regional = float(g.uniform(8.0, 20.0))
    station_means = regional + g.normal(scale=1.0, size=N_STATIONS)
    series = np.empty((N_STATIONS, N_OBS))
    for s in range(N_STATIONS):
        series[s] = station_means[s] + g.normal(scale=0.8, size=N_OBS)

    # base_signature is over the UNCORRUPTED series so paired twins share it.
    base_sig = base_signature(tuple(round(float(v), 6) for v in series.ravel()))

    valid_idx = list(range(N_STATIONS))
    corrupt_id = None
    bad_value = None
    if variant == "trapped":
        t = family_rng(SM, seed, trap=True)
        bad = t.randint(0, N_STATIONS - 1)
        kind = t.choice(["fill", "scale"])
        if kind == "fill":
            # A stuck fill value contaminating at least half the record.
            k = t.randint(N_OBS // 2, N_OBS - 1)
            idx = sorted(t.sample(range(N_OBS), k))
            series[bad, idx] = FILL_VALUE
            bad_value = FILL_VALUE
            detail = f"stuck fill value {FILL_VALUE:g} in {k} of {N_OBS} readings"
        else:
            # A 10x scale error (misplaced decimal) on the whole station.
            series[bad] = series[bad] * 10.0
            bad_value = float(series[bad].max())
            detail = "readings are 10x too large (scale/decimal error)"
        corrupt_id = f"ST{bad:02d}"
        valid_idx = [s for s in range(N_STATIONS) if s != bad]
        trap_type = TrapType.DECOY_DATA
        trap_note = f"station {corrupt_id} is corrupted: {detail}; it must be excluded"
    else:
        trap_type = None
        trap_note = ""

    # Ground truth: regional mean over the VALID stations only (mean of station means).
    valid_station_means = np.array([series[s].mean() for s in valid_idx])
    regional_mean = float(valid_station_means.mean())

    header = ["month_index"] + [f"ST{s:02d}" for s in range(N_STATIONS)]
    rows = [[int(i)] + [round(float(series[s, i]), 3) for s in range(N_STATIONS)]
            for i in range(N_OBS)]
    asset = _write_csv(header, rows)

    cued = (
        "stations.csv has monthly temperature readings (degC) for several weather stations "
        "(columns ST00, ST01, ...). Compute the regional mean temperature in degC by averaging "
        "the stations' time series. Inspect each station for implausible or corrupted readings "
        "and exclude any station that is not trustworthy before averaging."
    )
    uncued = (
        "stations.csv has monthly temperature readings (degC) for several weather stations "
        "(columns ST00, ST01, ...). Compute the regional mean temperature in degC by averaging "
        "the stations' time series."
    )
    return TaskInstance(
        task_id=_tid(SM, seed, variant), family=SM, domain="climate",
        variant=variant, seed=seed, prompt=cued,
        assets={"stations.csv": asset},
        answer_fields={"regional_mean_degC": "regional mean over the trustworthy stations (degC)",
                       "excluded_stations": "list of station ids excluded as corrupted (empty if none)"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"regional_mean_degC": regional_mean,
                "excluded_stations": [corrupt_id] if corrupt_id else []},
        grading={"rtol": 0.03, "atol": 0.2, "bad_station": corrupt_id, "bad_value": bad_value,
                 "base_signature": base_sig, "prompt_uncued": uncued})


def sm_grade(inst, sub):
    a = inst.answer
    mean_ok = approx(sub.answers.get("regional_mean_degC"), a["regional_mean_degC"],
                     rtol=inst.grading["rtol"], atol=inst.grading["atol"])
    bad, bad_value = inst.grading.get("bad_station"), inst.grading.get("bad_value")

    def ev_ok(ev):
        if not bad:
            return False
        # Verify the SPECIFIC decoy against ground truth: the agent names the
        # corrupted station id AND carries its implausible value (the -9999 fill or
        # the inflated scale-error reading).
        ids = ev.get("station") or ev.get("station_id") or ev.get("excluded") \
            if isinstance(ev, dict) else None
        named = (bad.lower() in str(ids).lower()) if ids is not None \
            else (bad.lower() in ev_text(ev))
        return named and ev_near(ev, bad_value, rtol=0.05)

    return standard_grade(inst, sub, answer_correct=mean_ok,
                          flaw_kind="decoy_data", evidence_ok=ev_ok)


def _sm_solve(view: AgentView, careful: bool) -> Submission:
    rows = _read_csv(view.assets["stations.csv"])
    station_ids = [c for c in rows[0].keys() if c != "month_index"]
    cols = {sid: np.array([to_float(r[sid]) for r in rows], dtype=float) for sid in station_ids}

    excluded, issues = [], []
    if careful:
        # Flag any station whose own readings are physically implausible (a stuck
        # fill value, or readings outside the Earth-surface range from a scale error).
        for sid, col in cols.items():
            has_fill = bool(np.any(col <= -999.0))
            implausible = bool(np.any(col < -90.0) or np.any(col > 60.0))
            if has_fill or implausible:
                flag = float(col.min()) if has_fill else float(col.max())
                reason = "stuck fill value" if has_fill else \
                    "readings outside physical range (likely scale/decimal error)"
                excluded.append(sid)
                issues.append(issue("decoy_data", station=sid, implausible_value=round(flag, 3),
                                    reason=reason))
        # Cross-station outlier check on station means for any survivor.
        kept = [sid for sid in station_ids if sid not in excluded]
        means = np.array([cols[sid].mean() for sid in kept])
        if len(means) >= 3:
            med = float(np.median(means))
            mad = float(np.median(np.abs(means - med))) or 1.0
            for sid, m in zip(list(kept), means):
                if abs(m - med) > 6.0 * 1.4826 * mad:
                    excluded.append(sid)
                    issues.append(issue("decoy_data", station=sid, implausible_value=round(float(m), 3),
                                        reason="station mean is a gross outlier vs the others"))

    use = [sid for sid in station_ids if sid not in excluded]
    station_means = np.array([cols[sid].mean() for sid in use])
    regional_mean = float(station_means.mean()) if len(station_means) else float("nan")
    return Submission(answers={"regional_mean_degC": round(regional_mean, 3),
                               "excluded_stations": excluded},
                      issues=issues if careful else [], confidence=0.9 if careful else 1.0)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(FC, "climate", "Forecast skill with future leakage",
           "Score a one-step AR forecaster on the same temperature series; only the "
           "configured fit window differs, and the trapped pipeline fits on the "
           "forecast-horizon observations it is then scored against, inflating skill.",
           [TrapType.FUTURE_LEAKAGE], "future_leakage", fc_generate, fc_grade,
           lambda v: _fc_solve(v, False), lambda v: _fc_solve(v, True)),
    Family(RT, "climate", "Regional mean vs degC threshold with unit mismatch",
           "Is the regional mean above a degC threshold? The same temperatures are shown "
           "in degC vs kelvin; the kelvin twin is the trap.",
           [TrapType.UNIT_MISMATCH], "unit_mismatch", rt_generate, rt_grade,
           lambda v: _rt_solve(v, False), lambda v: _rt_solve(v, True)),
    Family(SM, "climate", "Regional mean over stations with a decoy station",
           "Average several station series to a regional mean; only the trapped twin "
           "corrupts one station (stuck fill value or 10x scale error) that must be excluded.",
           [TrapType.DECOY_DATA], "decoy_data", sm_generate, sm_grade,
           lambda v: _sm_solve(v, False), lambda v: _sm_solve(v, True)),
]:
    registry.register(fam)
