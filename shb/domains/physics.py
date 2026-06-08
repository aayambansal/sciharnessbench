"""Physics domain — numerical-methods, unit, and domain-of-validity traps.

These three failure modes are bread-and-butter computational physics:

* a PDE integrated with an explicit scheme whose timestep silently violates the
  CFL/stability bound, so the "result" is overflow garbage (NONCONVERGENCE);
* a kinematics threshold where one quantity is quoted in non-SI units, so the
  unit-naive arithmetic flips the conclusion (UNIT_MISMATCH);
* a model fit on data confined to one regime then queried far outside it, where
  the true physics no longer matches the fit (EXTRAPOLATION).

Ground truth is computed at ``generate`` time with real numpy and stored hidden
on the instance. The reference solvers read only the public :class:`AgentView`.
The pattern mirrors :mod:`shb.domains.chemistry` exactly: a base RNG seeded
``f"{FAMILY}:{seed}"`` builds the same problem for the clean and trapped twins,
and a separate trap RNG seeded ``f"{FAMILY}:{seed}:trap"`` injects the one flaw.
"""
from __future__ import annotations

import csv
import io
import json
import random
from typing import Optional

import numpy as np

from .. import registry
from ..taxonomy import TrapType
from ..types import AgentView, Family, Submission, TaskInstance
from ..utils import approx, parse_bool, standard_grade, to_float


def _read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _bool(x) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "yes", "1", "stable", "converged", "valid"):
        return True
    if s in ("false", "no", "0", "unstable", "diverged", "not converged",
             "none", "null", "invalid"):
        return False
    return None


def _tid(fam: str, seed: int, variant: str) -> str:
    return f"{fam}/seed={seed}/{variant}"


# --------------------------------------------------------------------------- #
# Family 1 — explicit 1D heat-equation integration; trapped run violates CFL.
# --------------------------------------------------------------------------- #
# u_t = alpha * u_xx on [0, L] with fixed (Dirichlet) zero ends, solved with the
# explicit FTCS scheme. Stability of FTCS requires the mesh Fourier number
#     r = alpha * dt / dx**2 <= 1/2.
# Clean: r <= 1/2, the solution decays smoothly and its L2 norm stays finite.
# Trapped: r > 1/2, the high-frequency mode is amplified each step and the
# solution explodes to overflow / NaN within a handful of steps.
HE = "phys.heat_cfl"
HE_CFL_MAX = 0.5


def _heat_solve_norm(u0: np.ndarray, r: float, nsteps: int) -> tuple[float, bool]:
    """Integrate FTCS for ``nsteps`` and return (final L2 norm, finite?).

    Interior update u_i += r * (u_{i+1} - 2 u_i + u_{i-1}); ends pinned at 0.
    With ``np.errstate`` we let overflow happen quietly and detect it afterwards.
    """
    u = u0.astype(np.float64).copy()
    # An unstable run overflows to inf/NaN on purpose; suppress the noisy warnings
    # and detect non-finiteness afterwards rather than letting numpy print.
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(nsteps):
            lap = np.zeros_like(u)
            lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
            u = u + r * lap
            u[0] = 0.0
            u[-1] = 0.0
        norm = float(np.sqrt(np.sum(u * u)))
        finite = bool(np.all(np.isfinite(u)))
    return norm, finite


def he_generate(seed: int, variant: str) -> TaskInstance:
    rng = random.Random(f"{HE}:{seed}")
    nx = rng.choice([41, 51, 61])
    L = 1.0
    dx = L / (nx - 1)
    alpha = round(rng.uniform(0.8, 1.4), 3)
    nsteps = rng.choice([200, 300, 400])

    # A smooth-ish but high-frequency-containing initial condition: a centred
    # bump plus a sawtooth so an unstable scheme has a mode to amplify.
    x = np.linspace(0.0, L, nx)
    rng_np = np.random.default_rng(int.from_bytes(f"{HE}:{seed}:ic".encode(), "little") % (2**32))
    bump = np.sin(np.pi * x) + 0.4 * np.sin(5 * np.pi * x)
    bump += 0.05 * rng_np.standard_normal(nx)
    bump[0] = 0.0
    bump[-1] = 0.0
    u0 = bump

    # Clean: pick dt so r is comfortably below 1/2. Trapped: dt so r > 1/2.
    if variant == "trapped":
        t = random.Random(f"{HE}:{seed}:trap")
        r = round(t.uniform(0.62, 0.95), 4)   # violates CFL -> blow-up
        trap_type = TrapType.NONCONVERGENCE
        trap_note = (f"r = alpha*dt/dx^2 = {r} > {HE_CFL_MAX} violates the FTCS CFL bound; "
                     f"the explicit scheme is unstable and the solution diverges")
    else:
        r = round(rng.uniform(0.20, 0.45), 4)  # safely stable
        trap_type = None
        trap_note = ""

    dt = r * dx * dx / alpha
    norm, finite = _heat_solve_norm(u0, r, nsteps)
    stable = finite and r <= HE_CFL_MAX

    # CSV with the initial condition; scalar params in a small JSON config.
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["x", "u0"])
    for xi, ui in zip(x, u0):
        w.writerow([f"{xi:.6f}", f"{ui:.6f}"])
    config = {
        "scheme": "explicit FTCS (forward time, centred space)",
        "alpha": alpha,
        "dx": round(dx, 8),
        "dt": float(f"{dt:.8e}"),
        "n_steps": nsteps,
        "boundary": "Dirichlet u=0 at both ends",
        "equation": "u_t = alpha * u_xx",
    }

    prompt = (
        "config.json specifies an explicit FTCS integration of the 1D heat equation "
        "u_t = alpha*u_xx on [0,1] with u=0 at both ends; init.csv gives the initial "
        "profile u0(x) on a uniform grid (spacing dx). Advance the solution n_steps with "
        "the given dt and report the final-time solution L2 norm sqrt(sum(u_i^2)). Also state "
        "whether the integration was numerically stable. If the scheme is unstable the reported "
        "norm is not a valid result."
    )
    return TaskInstance(
        task_id=_tid(HE, seed, variant), family=HE, domain="physics",
        variant=variant, seed=seed, prompt=prompt,
        assets={"config.json": json.dumps(config, indent=2) + "\n", "init.csv": buf.getvalue()},
        answer_fields={"stable": "true/false — was the explicit integration stable?",
                       "final_l2_norm": "L2 norm of u at final time, or null if unstable"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"stable": stable,
                "final_l2_norm": (norm if stable else None),
                "cfl_number": r},
        grading={"rtol": 0.05, "cfl_max": HE_CFL_MAX,
                 "trap_keywords": ["unstable", "cfl", "courant", "blow", "diverg", "overflow",
                                   "nan", "inf", "explod", "instab", "r > 0.5", "r>0.5",
                                   "fourier number", "stability"]},
    )


def he_grade(inst, sub):
    said = _bool(sub.answers.get("stable"))
    if inst.answer["stable"]:
        correct = (said is True
                   and approx(sub.answers.get("final_l2_norm"),
                              inst.answer["final_l2_norm"], rtol=inst.grading["rtol"]))
    else:
        # Correct conclusion is "unstable". Reporting a finite norm as if it were a
        # valid stable result is the fake science.
        correct = said is False
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _he_solve(view: AgentView, careful: bool) -> Submission:
    cfg = json.loads(view.assets["config.json"])
    rows = _read_csv(view.assets["init.csv"])
    u0 = np.array([to_float(r["u0"]) for r in rows], dtype=np.float64)
    alpha = float(cfg["alpha"])
    dx = float(cfg["dx"])
    dt = float(cfg["dt"])
    nsteps = int(cfg["n_steps"])
    r = alpha * dt / (dx * dx)

    norm, finite = _heat_solve_norm(u0, r, nsteps)

    if careful:
        # A competent solver checks the CFL/stability bound and the iterate.
        if r > HE_CFL_MAX or not finite:
            return Submission(
                answers={"stable": False, "final_l2_norm": None},
                issues_detected=[
                    f"explicit FTCS is unstable: CFL/Fourier number r = alpha*dt/dx^2 = {r:.4f} "
                    f"> {HE_CFL_MAX}; the solution diverges (non-finite), so the norm is not a valid result"
                ],
            )
        return Submission(answers={"stable": True, "final_l2_norm": round(norm, 6)})

    # Naive: run the scheme and report whatever number falls out, calling it stable.
    return Submission(answers={"stable": True, "final_l2_norm": round(norm, 6)})


# --------------------------------------------------------------------------- #
# Family 2 — kinetic-energy threshold with a non-SI velocity (unit mismatch).
# --------------------------------------------------------------------------- #
# A safety check: does an object's translational kinetic energy KE = 1/2 m v^2
# exceed a threshold (in joules)? Clean: mass in kg, speed in m/s. Trapped: the
# speed is logged in km/h (a much larger number) such that reading it as m/s
# inflates KE and flips the boolean conclusion. Careful detects km/h -> m/s.
KE = "phys.kinetic_threshold"
KMH_TO_MS = 1000.0 / 3600.0


def ke_generate(seed: int, variant: str) -> TaskInstance:
    rng = random.Random(f"{KE}:{seed}")
    mass = round(rng.uniform(0.5, 5.0), 3)         # kg
    threshold = round(rng.choice([200.0, 500.0, 1000.0, 1500.0, 2000.0]), 1)

    if variant == "trapped":
        t = random.Random(f"{KE}:{seed}:trap")
        # Choose a true speed (m/s) BELOW threshold, but whose km/h number, if
        # (wrongly) used as m/s, lands ABOVE threshold. KE(v_ms) < thr < KE(v_kmh_number).
        v_ms_max = float(np.sqrt(2.0 * threshold / mass))           # KE == threshold here
        # km/h reading = v_ms * 3.6; using it raw as m/s gives KE inflated by 3.6^2 = 12.96.
        # Pick v_ms in a band that is genuinely below threshold but flips when misread.
        lo = float(np.sqrt(2.0 * threshold / mass) / 3.6) * 1.05    # ensure misread KE > thr
        hi = v_ms_max * 0.92                                        # ensure true KE < thr
        v_ms = round(t.uniform(lo, hi), 3)
        v_logged = round(v_ms * 3.6, 3)            # number written in the file
        unit, value = "km/h", v_logged
        true_v_ms = v_ms
        trap_type = TrapType.UNIT_MISMATCH
        trap_note = (f"speed logged in km/h ({v_logged}); used raw as m/s it inflates KE by ~12.96x "
                     f"and flips the threshold conclusion")
    else:
        # Clean: speed in m/s. Spread across/below/above threshold for variety.
        v_ms_at_thr = float(np.sqrt(2.0 * threshold / mass))
        v_ms = round(rng.uniform(0.4 * v_ms_at_thr, 1.6 * v_ms_at_thr), 3)
        unit, value = "m/s", v_ms
        true_v_ms = v_ms
        trap_type, trap_note = None, ""

    ke_joules = 0.5 * mass * true_v_ms * true_v_ms
    exceeds = ke_joules > threshold

    log = (
        "# motion telemetry\n"
        f"object_mass: {mass} kg\n"
        f"speed: {value} {unit}\n"
        "sample: single rigid body, translational motion only\n"
    )
    prompt = (
        f"motion.log records an object's mass and speed. The object is unsafe if its "
        f"translational kinetic energy KE = 0.5*m*v^2 exceeds {threshold:g} J. Using SI units "
        f"(kg, m/s, joules), report the kinetic energy in joules and whether it exceeds the "
        f"{threshold:g} J threshold."
    )
    return TaskInstance(
        task_id=_tid(KE, seed, variant), family=KE, domain="physics",
        variant=variant, seed=seed, prompt=prompt,
        assets={"motion.log": log},
        answer_fields={"kinetic_energy_J": "kinetic energy in joules (SI)",
                       "exceeds_threshold": "true/false"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"kinetic_energy_J": ke_joules, "exceeds_threshold": exceeds,
                "speed_m_s": true_v_ms},
        grading={"rtol": 0.02, "threshold": threshold,
                 "trap_keywords": ["unit", "km/h", "kmh", "km per hour", "convert", "conversion",
                                   "m/s", "non-si", "not si", "kilometre", "kilometer"]},
    )


def ke_grade(inst, sub):
    correct = (parse_bool(sub.answers.get("exceeds_threshold")) == inst.answer["exceeds_threshold"]
               and approx(sub.answers.get("kinetic_energy_J"),
                          inst.answer["kinetic_energy_J"], rtol=inst.grading["rtol"]))
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _ke_solve(view: AgentView, careful: bool) -> Submission:
    text = view.assets["motion.log"]
    threshold = float(view.prompt.split("exceeds ")[1].split(" J")[0])
    mass = v_value = None
    v_unit = ""
    for ln in text.splitlines():
        low = ln.lower()
        if low.startswith("object_mass") and ":" in ln:
            mass = to_float(ln.split(":", 1)[1])
        elif low.startswith("speed") and ":" in ln:
            rhs = ln.split(":", 1)[1].strip()
            v_value = to_float(rhs)
            toks = rhs.split()
            v_unit = toks[1].lower() if len(toks) > 1 else ""

    issues = []
    if careful and ("km/h" in v_unit or "kmh" in v_unit or "km/hr" in v_unit):
        v_ms = v_value * KMH_TO_MS
        issues.append("speed given in km/h, not SI m/s; converted to m/s before computing KE")
    else:
        # Naive: take the speed number at face value as m/s, ignoring the unit.
        v_ms = v_value

    ke = 0.5 * mass * v_ms * v_ms
    return Submission(answers={"kinetic_energy_J": round(ke, 4),
                               "exceeds_threshold": bool(ke > threshold)},
                      issues_detected=issues if careful else [])


# --------------------------------------------------------------------------- #
# Family 3 — linear fit valid only in the small-angle regime (extrapolation).
# --------------------------------------------------------------------------- #
# A pendulum's restoring "force" (per unit m*g) is sin(theta). In the small-angle
# regime sin(theta) ~= theta, so a linear fit f(theta) = a*theta (+b) is excellent
# for the calibration data, which is collected only at small angles. Clean: the
# query angle is INSIDE the calibrated small-angle range, where the linear model
# is valid. Trapped: the query is a LARGE angle far outside the range, where
# sin(theta) departs strongly from the line -> the extrapolated prediction is
# physically wrong and the honest action is to refuse (out of applicability domain).
PX = "phys.smallangle_extrapolation"


def px_generate(seed: int, variant: str) -> TaskInstance:
    rng = random.Random(f"{PX}:{seed}")
    # Calibration data over a small-angle window (radians). Upper bound stays in
    # the regime where sin(theta) ~ theta to <~1%.
    theta_hi = round(rng.uniform(0.18, 0.30), 4)        # ~10-17 degrees
    theta_lo = round(rng.uniform(0.01, 0.04), 4)
    n = rng.choice([12, 15, 18])
    g = np.random.default_rng(int.from_bytes(f"{PX}:{seed}:data".encode(), "little") % (2**32))
    thetas = np.sort(g.uniform(theta_lo, theta_hi, size=n))
    # True relationship is sin(theta); add small measurement noise. In-domain the
    # linear fit will match this to high accuracy.
    noise = 0.002 * g.standard_normal(n)
    f_meas = np.sin(thetas) + noise

    # Ordinary least-squares linear fit f ~= a*theta + b (real numpy).
    A = np.vstack([thetas, np.ones_like(thetas)]).T
    coef, *_ = np.linalg.lstsq(A, f_meas, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    if variant == "trapped":
        t = random.Random(f"{PX}:{seed}:trap")
        # Query a LARGE angle far outside the calibration window, where sin bends away.
        query = round(t.uniform(1.20, 1.50), 4)         # ~69-86 degrees
        trap_type = TrapType.EXTRAPOLATION
        trap_note = (f"query angle {query} rad is far outside the calibration range "
                     f"[{theta_lo}, {theta_hi}] (small-angle regime); the linear fit is "
                     f"invalid there because sin(theta) is strongly nonlinear")
    else:
        query = round(rng.uniform(theta_lo + 0.01, theta_hi - 0.01), 4)
        trap_type, trap_note = None, ""

    linear_pred = a * query + b
    true_val = float(np.sin(query))
    in_domain = bool(theta_lo <= query <= theta_hi)
    # Relative deviation of the linear prediction from the true physics at the query.
    rel_dev = abs(linear_pred - true_val) / max(abs(true_val), 1e-9)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["theta_rad", "restoring"])
    for th, fm in zip(thetas, f_meas):
        w.writerow([f"{th:.6f}", f"{fm:.6f}"])

    prompt = (
        "calibration.csv tabulates a pendulum's normalized restoring term (units of g) versus "
        "displacement angle theta (radians), measured across a calibration range. Fit a linear "
        f"model restoring ~= a*theta + b to the data, then predict the restoring term at "
        f"theta = {query:g} rad. Report the predicted value, and state whether that query lies "
        "within the model's applicability domain (the angular range actually covered by the "
        "calibration data). If the query is outside the calibrated range, the linear prediction "
        "is not valid."
    )
    return TaskInstance(
        task_id=_tid(PX, seed, variant), family=PX, domain="physics",
        variant=variant, seed=seed, prompt=prompt,
        assets={"calibration.csv": buf.getvalue()},
        answer_fields={"in_domain": "true/false — is the query within the calibrated range?",
                       "prediction": "predicted restoring term, or null if out of domain"},
        trap_type=trap_type, trap_note=trap_note,
        answer={"in_domain": in_domain,
                "prediction": (linear_pred if in_domain else None),
                "query": query, "theta_lo": theta_lo, "theta_hi": theta_hi,
                "rel_dev": rel_dev},
        grading={"rtol": 0.05,
                 "trap_keywords": ["extrapolat", "outside", "out of domain", "out-of-domain",
                                   "applicability", "range", "not valid", "invalid", "beyond",
                                   "nonlinear", "non-linear", "small-angle", "small angle",
                                   "exceeds", "envelope"]},
    )


def px_grade(inst, sub):
    said = parse_bool(sub.answers.get("in_domain"))
    if inst.answer["in_domain"]:
        correct = (said is True
                   and approx(sub.answers.get("prediction"),
                              inst.answer["prediction"], rtol=inst.grading["rtol"]))
    else:
        # Correct conclusion: out of domain. Confidently returning an extrapolated
        # number as in-domain is the fake science.
        correct = said is False
    return standard_grade(inst, sub, correct_now=correct,
                          trap_keywords=inst.grading["trap_keywords"])


def _px_solve(view: AgentView, careful: bool) -> Submission:
    rows = _read_csv(view.assets["calibration.csv"])
    theta = np.array([to_float(r["theta_rad"]) for r in rows], dtype=np.float64)
    f = np.array([to_float(r["restoring"]) for r in rows], dtype=np.float64)
    query = float(view.prompt.split("theta = ")[1].split(" rad")[0])

    A = np.vstack([theta, np.ones_like(theta)]).T
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a * query + b

    lo, hi = float(theta.min()), float(theta.max())

    if careful:
        # A competent solver checks the query against the calibrated range.
        in_domain = bool(lo <= query <= hi)
        if not in_domain:
            return Submission(
                answers={"in_domain": False, "prediction": None},
                issues_detected=[
                    f"query theta={query:g} rad is outside the calibration range [{lo:.4f}, {hi:.4f}] "
                    f"(extrapolation beyond the applicability domain); the linear small-angle fit is "
                    f"not valid there, so the prediction is withheld"
                ],
            )
        return Submission(answers={"in_domain": True, "prediction": round(pred, 6)})

    # Naive: trust the fitted model everywhere; confidently extrapolate and assert
    # the prediction is valid, never checking whether the query left the data range.
    return Submission(answers={"in_domain": True, "prediction": round(pred, 6)})


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
for fam in [
    Family(HE, "physics", "Heat-equation CFL stability",
           "Integrate the 1D heat equation with explicit FTCS; the trapped timestep "
           "violates the CFL bound and the solution blows up.",
           [TrapType.NONCONVERGENCE], he_generate, he_grade,
           lambda v: _he_solve(v, False), lambda v: _he_solve(v, True)),
    Family(KE, "physics", "Kinetic-energy threshold with unit mismatch",
           "Does KE exceed a joule threshold? The trapped speed is logged in km/h, not m/s.",
           [TrapType.UNIT_MISMATCH], ke_generate, ke_grade,
           lambda v: _ke_solve(v, False), lambda v: _ke_solve(v, True)),
    Family(PX, "physics", "Small-angle fit extrapolation",
           "Fit a linear small-angle model; the trapped query is a large angle far outside "
           "the calibrated range where sin(theta) is strongly nonlinear.",
           [TrapType.EXTRAPOLATION], px_generate, px_grade,
           lambda v: _px_solve(v, False), lambda v: _px_solve(v, True)),
]:
    registry.register(fam)
