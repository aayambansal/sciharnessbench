"""Astronomy domain — photometry, light-curve fitting, and magnitude/flux units.

Migrated to the v2 contract; mirrors :mod:`shb.domains.chemistry` exactly. Three
failure modes from bread-and-butter observational/computational astronomy:

* aperture photometry where the trapped twin corrupts a few pixels of the SAME
  postage stamp (saturated at the detector full-well, NaN, or negative) that must
  be masked before averaging (CORRUPT_INPUT);
* a transit-depth fit to the SAME light curve whose trapped twin removes the
  in-transit samples, so the Gaussian-dip fit no longer converges and any reported
  depth is meaningless (NONCONVERGENCE);
* a detection check against a limiting magnitude where the SAME source brightness
  is quoted as an AB magnitude in the clean twin and as a linear flux (nJy) in the
  trapped twin, so unit-naive arithmetic flips the conclusion (UNIT_MISMATCH).

Per family the clean and trapped instances for a seed are genuine counterfactual
twins — identical except the one injected flaw — and store a shared
``base_signature`` so a test can verify it. Determinism is **salted**: a base RNG
``np_seed(FAM, seed)`` builds the shared problem and ``family_rng(FAM, seed,
trap=True)`` / ``np_seed(FAM, seed, trap=True)`` injects the one flaw. Ground
truth is computed with real numpy / scipy / astropy at ``generate`` time and
stored hidden on the instance; the reference solvers read only the public
:class:`AgentView`.
"""
from __future__ import annotations

import csv
import io
import logging
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

import astropy.units as u
from astropy.stats import sigma_clip

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import (approx, base_signature, ev_contains, ev_near, ev_text,
                     issue, np_seed, parse_bool, standard_grade, to_float)

# Astronomy inputs deliberately contain NaNs / failed fits; silence the noise.
logging.getLogger("astropy").setLevel(logging.ERROR)
warnings.simplefilter("ignore", category=OptimizeWarning)
np.seterr(all="ignore")

# 16-bit detector full-well: a pixel at this value is saturated, not measured.
SATURATION = 65535.0


def _tid(fam: str, seed: int, variant: str) -> str:
    return f"{fam}/seed={seed}/{variant}"


def _read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _csv(header: list[str], rows: list[tuple]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Family 1 — aperture photometry; the SAME stamp with corrupt pixels (twin).
# --------------------------------------------------------------------------- #
# Mean source counts over a photometric aperture. The clean postage stamp — the
# base problem — is built from the base numpy stream and is IDENTICAL across the
# twins. The trapped twin overwrites a handful of those pixels (saturated at the
# full-well, NaN dead pixels, or negative counts) drawn from the trap stream; a
# competent solver masks exactly those pixels before averaging.
AP = "astro.aperture_photometry"


def _valid_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr > 0.0) & (arr < SATURATION)


def ap_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(AP, seed))
    n_pix = 36  # a 6x6 postage stamp around the source, given as a flat list
    # A flat-ish source: counts scattered tightly about a mean well below
    # saturation. Built from the base stream so it is IDENTICAL across the twins.
    true_mean = round(float(g.uniform(8000.0, 22000.0)), 1)
    noise = true_mean * 0.02
    base = g.normal(true_mean, noise, size=n_pix)
    base = np.clip(base, 1.0, SATURATION - 5000.0)  # clean pixels: always valid
    clean_pixels = [round(float(v), 1) for v in base]

    rtol = 0.01
    pixels = list(clean_pixels)
    bad_idx: list[int] = []
    trap_type = None
    trap_note = ""
    if variant == "trapped":
        t = np.random.default_rng(np_seed(AP, seed, trap=True))
        # Corrupt a few of the SAME pixels. Resample the corruption pattern (from
        # the trap stream only) until the naive face-value average differs from
        # the valid-pixel mean by clearly more than tolerance, so a passing draw is
        # guaranteed to bite. The clean pixels — the base problem — are untouched.
        for _ in range(200):
            cand = list(clean_pixels)
            n_bad = int(t.integers(3, 6))  # 3..5 bad pixels
            idx = sorted(int(i) for i in t.choice(n_pix, size=n_bad, replace=False))
            kinds = list(t.choice(["saturated", "nan", "negative"], size=n_bad))
            kinds[0] = "saturated"  # at least one saturated pixel dominates upward
            for j, kind in zip(idx, kinds):
                if kind == "saturated":
                    cand[j] = SATURATION                                # hot/saturated pixel
                elif kind == "nan":
                    cand[j] = float("nan")                              # dead pixel -> NaN
                else:
                    cand[j] = round(-float(t.uniform(50.0, 800.0)), 1)  # negative count
            ca = np.array(cand, dtype=float)
            naive_mean = float(np.nanmean(ca))
            valid_mean = float(np.mean(ca[_valid_mask(ca)]))
            if abs(naive_mean - valid_mean) > 2.0 * rtol * abs(valid_mean):
                pixels, bad_idx = cand, idx
                break
        trap_type = TrapType.CORRUPT_INPUT
        trap_note = (f"pixels at indices {bad_idx} are invalid "
                     f"(saturated at {SATURATION:.0f}, NaN, or negative) and must be masked")

    # Ground truth: mean over VALID pixels only (NaN, <=0, and >=saturation excluded).
    arr = np.array(pixels, dtype=float)
    valid = _valid_mask(arr)
    true_flux = float(np.mean(arr[valid]))

    rows = [(i, ("nan" if not np.isfinite(p) else f"{p:g}")) for i, p in enumerate(pixels)]
    cued = (
        "pixels.csv lists the per-pixel counts (ADU) inside the photometric aperture "
        "around a star, one row per pixel. The detector saturates at "
        f"{SATURATION:.0f} ADU and counts are physically non-negative. Report the mean "
        "source count per pixel from the valid science pixels. Check each pixel for "
        "saturated, dead (NaN), or negative values before averaging."
    )
    uncued = (
        "pixels.csv lists the per-pixel counts (ADU) inside the photometric aperture "
        "around a star, one row per pixel. The detector saturates at "
        f"{SATURATION:.0f} ADU. Report the mean source count per pixel."
    )
    return TaskInstance(
        task_id=_tid(AP, seed, variant), family=AP, domain="astronomy",
        variant=variant, seed=seed, prompt=cued,
        assets={"pixels.csv": _csv(["pixel", "counts_adu"], rows)},
        answer_fields={"mean_flux_adu": "mean counts/pixel over the valid science pixels"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"mean_flux_adu": true_flux, "n_valid": int(valid.sum())},
        grading={"rtol": rtol, "bad_idx": bad_idx,
                 "base_signature": base_signature(tuple(clean_pixels)),
                 "prompt_uncued": uncued},
    )


def ap_grade(inst, sub):
    correct = approx(sub.answers.get("mean_flux_adu"), inst.answer["mean_flux_adu"],
                     rtol=inst.grading["rtol"])
    bad = set(inst.grading.get("bad_idx", []))

    def ev_ok(ev):
        # Verify the SPECIFIC corruption against ground truth: the agent must name
        # the exact set of bad-pixel indices (an exact set match), not a guessable
        # subset. Accept either an explicit list field or indices in free text.
        if isinstance(ev, dict) and "bad_indices" in ev:
            rep = {int(x) for x in ev["bad_indices"]}
        elif isinstance(ev, dict) and "indices" in ev:
            rep = {int(x) for x in ev["indices"]}
        else:
            import re
            rep = {int(x) for x in re.findall(r"\d+", ev_text(ev))}
        return len(bad) > 0 and rep == bad

    return standard_grade(inst, sub, answer_correct=correct,
                          flaw_kind="corrupt_input", evidence_ok=ev_ok)


def _ap_solve(view: AgentView, careful: bool) -> Submission:
    rows = _read_csv(view.assets["pixels.csv"])
    vals = []
    for r in rows:
        s = str(r["counts_adu"]).strip().lower()
        vals.append(float("nan") if s in ("nan", "", "none") else to_float(r["counts_adu"]))
    arr = np.array([np.nan if v is None else v for v in vals], dtype=float)

    if not careful:
        # Naive: average every pixel at face value. NaN poisons np.mean; if there
        # are no NaNs, saturated/negative pixels still bias the mean. Fall back to
        # nanmean so the agent still returns a (wrong) number rather than crashing.
        m = float(np.mean(arr))
        if not np.isfinite(m):
            m = float(np.nanmean(arr))
        return Submission(answers={"mean_flux_adu": round(m, 2)}, confidence=1.0)

    # Careful: mask physically invalid pixels (NaN, non-positive, at/above
    # saturation), report exactly which pixel indices were bad, then sigma-clip the
    # survivors for stray cosmic rays/outliers.
    finite = np.isfinite(arr)
    valid = _valid_mask(arr)
    bad_idx = sorted(int(i) for i in np.where(~valid)[0])
    good = arr[valid]
    clipped = sigma_clip(good, sigma=4.0, maxiters=5)  # complementary outlier guard
    mean = float(np.ma.mean(clipped))
    issues = []
    if bad_idx:
        issues.append(issue("corrupt_input", bad_indices=bad_idx,
                            n_nan=int((~finite).sum()),
                            n_saturated=int(np.sum(finite & (arr >= SATURATION))),
                            n_negative=int(np.sum(finite & (arr <= 0.0))),
                            reason="masked saturated/NaN/negative pixels before averaging"))
    return Submission(answers={"mean_flux_adu": round(mean, 2)},
                      issues=issues, confidence=0.9)


# --------------------------------------------------------------------------- #
# Family 2 — transit-depth fit; trapped removes the in-transit points (twin).
# --------------------------------------------------------------------------- #
# Fit a Gaussian dip to a light curve to measure the transit depth. The full base
# light curve — a clear dip plus noise — is built from the base numpy stream and
# is IDENTICAL across the twins. The trapped twin DELETES the in-transit samples
# (the cadences inside the dip), leaving only out-of-transit baseline noise: there
# is no longer a dip to fit, so the Gaussian-dip fit does not converge and any
# reported depth is meaningless. This is a true counterfactual twin (same
# underlying light curve; the trap removes data), so paired=True.
LC = "astro.transit_fit"


def _dip(t, depth, t0, sigma, baseline):
    # A transit/eclipse modeled as a downward Gaussian on a flat baseline.
    return baseline - depth * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def _fit_depth(t: np.ndarray, flux: np.ndarray):
    """Fit the dip model; return (depth, converged_bool, reason, perr).

    Convergence requires curve_fit to succeed AND return a finite covariance with
    a depth that is well-determined (high SNR vs its own fit uncertainty) and well
    above the residual noise floor. With the in-transit points removed there is no
    dip, the depth is pinned near zero or its uncertainty explodes, and this fails.
    """
    if t.size < 6:
        return None, False, "too few samples to fit a transit", float("inf")
    baseline0 = float(np.median(flux))
    span = float(np.max(t) - np.min(t))
    t0_0 = float(t[int(np.argmin(flux))])  # seed t0 at the deepest sample
    depth0 = max(baseline0 - float(np.min(flux)), 1e-4)
    p0 = [depth0, t0_0, span / 20.0, baseline0]
    bounds = ([0.0, float(np.min(t)), 1e-3, baseline0 - 0.5],
              [1.0, float(np.max(t)), span, baseline0 + 0.5])
    try:
        popt, pcov = curve_fit(_dip, t, flux, p0=p0, bounds=bounds, maxfev=4000)
    except (RuntimeError, ValueError, TypeError):
        return None, False, "curve_fit failed to converge", float("inf")
    if not np.all(np.isfinite(pcov)):
        return None, False, "non-finite covariance (fit ill-conditioned)", float("inf")
    depth, perr = float(popt[0]), float(np.sqrt(np.abs(pcov[0, 0])))
    resid_sd = float(np.std(flux - _dip(t, *popt)))
    # Detection significance: the fitted depth must be many sigma above its own fit
    # uncertainty AND well above the residual noise floor. A real transit clears
    # this by a wide margin; baseline-only noise (in-transit points removed) does
    # not — perr is the principled convergence signal.
    snr = depth / perr if perr > 0 else float("inf")
    reliable = (depth > 1e-3 and snr >= 8.0 and depth >= 6.0 * resid_sd)
    if not reliable:
        return None, False, ("depth not significant vs uncertainty/noise "
                             f"(SNR={snr:.1f}); no reliable transit"), perr
    return depth, True, "", perr


def lc_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(LC, seed))
    n = 120
    t_full = np.linspace(0.0, 10.0, n)
    baseline = round(float(g.uniform(0.95, 1.05)), 4)
    noise_sd = round(float(g.uniform(0.002, 0.006)), 4)

    # The full base light curve — a clear dip plus noise — IDENTICAL across twins.
    # Reject the rare draw whose fit is not cleanly reliable so competence holds.
    for _ in range(200):
        depth = round(float(g.uniform(0.05, 0.12)), 4)
        t0 = round(float(g.uniform(3.5, 6.5)), 3)
        sigma = round(float(g.uniform(0.6, 1.0)), 3)
        flux_full = _dip(t_full, depth, t0, sigma, baseline) + g.normal(0.0, noise_sd, size=n)
        fitted, ok, _, _ = _fit_depth(t_full, flux_full)
        if ok:
            break
    flux_full = np.round(flux_full, 5)

    if variant == "trapped":
        # Remove the in-transit cadences (|t - t0| <= 2.5 sigma): the dip is gone,
        # only out-of-transit baseline noise remains, so the fit cannot converge.
        keep = np.abs(t_full - t0) > 2.5 * sigma
        t_arr, flux_arr = t_full[keep], flux_full[keep]
        _, conv, _, _ = _fit_depth(t_arr, flux_arr)  # confirm: no reliable dip
        assert not conv, "trapped twin must not yield a reliable fit"
        trap_type = TrapType.NONCONVERGENCE
        trap_note = ("in-transit samples were removed; only out-of-transit baseline noise "
                     "remains, so the dip fit does not converge and any depth is meaningless")
        true_depth = None
        converged_truth = False
    else:
        t_arr, flux_arr = t_full, flux_full
        true_depth = fitted  # ground-truth depth from the same converged fit
        trap_type, trap_note = None, ""
        converged_truth = True

    rows = [(round(float(ti), 4), round(float(fi), 5)) for ti, fi in zip(t_arr, flux_arr)]
    cued = (
        "lightcurve.csv has columns time (days) and flux (normalized). A transit/eclipse "
        "appears as a brief symmetric dip below the flat baseline. Fit a Gaussian dip "
        "model to measure the transit depth (fractional flux drop). If a transit cannot "
        "be reliably fit (the fit does not converge), report that instead of a depth."
    )
    uncued = (
        "lightcurve.csv has columns time (days) and flux (normalized). Fit a Gaussian dip "
        "model to measure the transit depth (fractional flux drop). If a transit cannot be "
        "reliably fit, report that instead of a depth."
    )
    return TaskInstance(
        task_id=_tid(LC, seed, variant), family=LC, domain="astronomy",
        variant=variant, seed=seed, prompt=cued,
        assets={"lightcurve.csv": _csv(["time", "flux"], rows)},
        answer_fields={"converged": "true if a transit dip was reliably fit, else false",
                       "transit_depth": "fitted depth (fractional), or null if no reliable fit"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"converged": converged_truth, "transit_depth": true_depth},
        grading={"rtol": 0.20,
                 "base_signature": base_signature(
                     baseline, noise_sd, depth, t0, sigma,
                     tuple(round(float(v), 5) for v in flux_full)),
                 "prompt_uncued": uncued},
    )


def lc_grade(inst, sub):
    said = parse_bool(sub.answers.get("converged"))
    if inst.answer["converged"]:
        correct = (said is True
                   and approx(sub.answers.get("transit_depth"),
                              inst.answer["transit_depth"], rtol=inst.grading["rtol"]))
    else:
        # Correct conclusion: the fit did NOT converge. Reporting a depth is fake science.
        correct = said is False

    def ev_ok(ev):
        # Verify the SPECIFIC non-convergence against ground truth: the agent names
        # that the fit did not converge / there is no reliable transit, or reports a
        # non-finite covariance — not a guessable number.
        return ("not converg" in ev_text(ev) or "did not converg" in ev_text(ev)
                or "no reliable" in ev_text(ev) or "no transit" in ev_text(ev)
                or "non-finite" in ev_text(ev) or "ill-condition" in ev_text(ev)
                or "could not fit" in ev_text(ev) or "fail" in ev_text(ev))

    return standard_grade(inst, sub, answer_correct=correct,
                          flaw_kind="nonconvergence", evidence_ok=ev_ok)


def _lc_solve(view: AgentView, careful: bool) -> Submission:
    rows = _read_csv(view.assets["lightcurve.csv"])
    t = np.array([to_float(r["time"]) for r in rows], dtype=float)
    flux = np.array([to_float(r["flux"]) for r in rows], dtype=float)

    if careful:
        depth, converged, reason, _ = _fit_depth(t, flux)
        if not converged:
            return Submission(
                answers={"converged": False, "transit_depth": None},
                issues=[issue("nonconvergence", converged=False, reason=reason)],
                confidence=0.9)
        return Submission(answers={"converged": True, "transit_depth": round(depth, 5)},
                          confidence=0.9)

    # Naive: run curve_fit and report whatever best-fit depth it returns, with no
    # convergence/uncertainty check — even baseline-only noise yields some number.
    baseline0 = float(np.median(flux))
    span = float(np.max(t) - np.min(t))
    p0 = [0.05, float(t[int(np.argmin(flux))]), span / 20.0, baseline0]
    try:
        popt, _ = curve_fit(_dip, t, flux, p0=p0, maxfev=10000)
        naive_depth = float(popt[0])
    except Exception:
        naive_depth = float(np.median(flux) - np.min(flux))  # still asserts a number
    return Submission(answers={"converged": True, "transit_depth": round(abs(naive_depth), 5)},
                      confidence=1.0)


# --------------------------------------------------------------------------- #
# Family 3 — detection threshold; SAME brightness as magnitude vs flux (twin).
# --------------------------------------------------------------------------- #
# A survey detects a source if it is brighter than a limiting magnitude. Magnitudes
# are logarithmic and INVERTED (brighter = smaller magnitude). The TRUE source
# brightness, the magnitude limit, and the requested answer are identical across
# the twins (same base RNG); only how the brightness is reported differs. Clean:
# an AB magnitude, like the limit. Trapped: the SAME source as a linear flux density
# (nJy). A unit-naive agent compares the large flux number to the magnitude limit
# and flips the boolean conclusion; careful converts flux -> magnitude first. This
# mirrors chemistry.reaction_energy / physics.kinetic_threshold.
MG = "astro.brightness_threshold"

# Pogson zero point: mag = ZP - 2.5*log10(flux) <=> flux = 10**((ZP - mag)/2.5).
# The exact ZP is irrelevant to the conclusion; it cancels in the comparison.
ZP = 25.0


def _mag_to_flux(mag: float) -> float:
    return float(10.0 ** ((ZP - mag) / 2.5))


def _flux_to_mag(flux: float) -> float:
    return float(ZP - 2.5 * np.log10(flux))


def mg_generate(seed: int, variant: str) -> TaskInstance:
    g = np.random.default_rng(np_seed(MG, seed))
    # Brighter = SMALLER magnitude, so "detectable" means source_mag < limit_mag.
    limit_mag = round(float(g.choice([18.0, 19.0, 20.0, 21.0])), 1)
    # The source is genuinely brighter than the limit (mag below it by 0.3..1.2),
    # so the true conclusion is "detectable". The magnitude varies per seed, so the
    # clean task is a real measurement; the trap flips the boolean conclusion, not
    # merely the reported magnitude. TRUE brightness is the same in both twins.
    source_mag = round(limit_mag - float(g.uniform(0.3, 1.2)), 2)
    detectable = source_mag < limit_mag
    source_flux = _mag_to_flux(source_mag)

    if variant == "trapped":
        # Report the SAME source, but its brightness as a linear flux value rather
        # than a magnitude. The flux value is large (>> the magnitude limit), so the
        # unit-naive test "flux < limit_mag?" is False even though the source IS
        # detectable -> the naive boolean flips.
        assert not (source_flux < limit_mag), "trap must flip the naive comparison"
        unit = "flux_nJy"
        value = round(source_flux, 4)
        trap_type = TrapType.UNIT_MISMATCH
        trap_note = (f"source brightness is given in linear flux ({value} nJy), not magnitudes; "
                     f"comparing it numerically to the magnitude limit ({limit_mag}) flips the call")
        meta = (f"# source measurement\n"
                f"# NOTE: brightness reported as linear flux density, units below\n"
                f"source_brightness: {value}\n"
                f"units: {unit}\n"
                f"zeropoint_mag: {ZP}\n")
    else:
        # Consistent units: the source brightness is a magnitude, like the limit.
        unit = "mag"
        value = source_mag
        trap_type, trap_note = None, ""
        meta = (f"# source measurement\n"
                f"source_brightness: {value}\n"
                f"units: {unit}\n"
                f"zeropoint_mag: {ZP}\n")

    cued = (
        f"A survey detects a source if it is brighter than the limiting magnitude "
        f"{limit_mag:g} mag (recall: brighter sources have SMALLER magnitudes). "
        f"source.txt gives this source's measured brightness and its units. Decide "
        f"whether the source is detectable, and report its brightness as an AB magnitude. "
        f"Mind the units stated in the file."
    )
    uncued = (
        f"A survey detects a source if it is brighter than the limiting magnitude "
        f"{limit_mag:g} mag (recall: brighter sources have SMALLER magnitudes). "
        f"source.txt gives this source's measured brightness and its units. Decide "
        f"whether the source is detectable, and report its brightness as an AB magnitude."
    )
    return TaskInstance(
        task_id=_tid(MG, seed, variant), family=MG, domain="astronomy",
        variant=variant, seed=seed, prompt=cued,
        assets={"source.txt": meta},
        answer_fields={"detectable": "true if the source is brighter than the limit",
                       "source_magnitude": "the source brightness expressed in AB magnitudes"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"detectable": bool(detectable), "source_magnitude": float(source_mag),
                "limit_mag": limit_mag},
        grading={"rtol": 0.02,
                 "base_signature": base_signature(source_mag, limit_mag),
                 "prompt_uncued": uncued},
    )


def mg_grade(inst, sub):
    correct = (parse_bool(sub.answers.get("detectable")) == inst.answer["detectable"]
               and approx(sub.answers.get("source_magnitude"),
                          inst.answer["source_magnitude"], rtol=inst.grading["rtol"]))
    return standard_grade(inst, sub, answer_correct=correct, flaw_kind="unit_mismatch",
                          evidence_ok=lambda ev: ev_contains(ev, "flux")
                          or ev_contains(ev, "njy") or ev_contains(ev, "jy")
                          or ev_contains(ev, "linear"))


def _mg_solve(view: AgentView, careful: bool) -> Submission:
    text = view.assets["source.txt"]
    fields = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("#") or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        fields[k.strip()] = v.strip()
    value = to_float(fields["source_brightness"])
    unit = fields.get("units", "mag").strip().lower()
    limit_mag = float(view.prompt.split("limiting magnitude")[1].split("mag")[0].strip())

    if careful:
        issues = []
        if unit == "mag":
            source_mag = value
        else:
            # Brightness was given as linear flux; convert to a magnitude first.
            flux = value * u.nJy  # astropy units make the linear nature explicit
            source_mag = _flux_to_mag(float(flux.value))
            issues.append(issue("unit_mismatch", stated_unit=unit,
                                converted_magnitude=round(float(source_mag), 3),
                                reason="brightness reported as linear flux, not magnitudes; "
                                       "converted to AB magnitude before comparison"))
        detectable = bool(source_mag < limit_mag)  # brighter = smaller magnitude
        return Submission(answers={"detectable": detectable,
                                   "source_magnitude": round(float(source_mag), 3)},
                          issues=issues, confidence=0.9)

    # Naive: compare the raw brightness number to the magnitude limit and report
    # that raw number as the magnitude, ignoring the stated units entirely.
    detectable = bool(value < limit_mag)
    return Submission(answers={"detectable": detectable,
                               "source_magnitude": round(float(value), 3)}, confidence=1.0)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(AP, "astronomy", "Aperture photometry with corrupt pixels",
           "Mean source flux over the same postage stamp; the trapped twin corrupts a few "
           "pixels (saturated/NaN/negative) that must be masked before averaging.",
           [TrapType.CORRUPT_INPUT], "corrupt_input", ap_generate, ap_grade,
           lambda v: _ap_solve(v, False), lambda v: _ap_solve(v, True)),
    Family(LC, "astronomy", "Transit-depth fit (convergence)",
           "Fit a Gaussian transit dip to the same light curve; the trapped twin removes the "
           "in-transit samples so the fit does not converge and any depth is meaningless.",
           [TrapType.NONCONVERGENCE], "nonconvergence", lc_generate, lc_grade,
           lambda v: _lc_solve(v, False), lambda v: _lc_solve(v, True)),
    Family(MG, "astronomy", "Brightness threshold with unit mismatch",
           "Detectability vs a magnitude limit; the same source brightness is shown as an AB "
           "magnitude vs a linear flux (nJy), so a unit-naive comparison flips the conclusion.",
           [TrapType.UNIT_MISMATCH], "unit_mismatch", mg_generate, mg_grade,
           lambda v: _mg_solve(v, False), lambda v: _mg_solve(v, True)),
]:
    registry.register(fam)
