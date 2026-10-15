# Pulse-Reverse Electrodeposition Analysis (`plot_deltaE.py`)

Automated per-pulse analysis of BioLogic EC-Lab `.mpr` files from pulse-reverse
copper electrodeposition experiments, built to quantify how stir-bar convection
affects deposition stability.

Processes raw potentiostat binaries into per-pulse ΔE measurements, detects the
onset of mass-transport collapse, and produces cross-condition comparison plots —
turning a proprietary instrument format into analysis-ready data with no manual
export step.

---

## Background

In pulse-reverse electrodeposition (PRED), current alternates between a cathodic
plating pulse and a brief anodic reverse pulse. Sustaining a high cathodic current
density requires Cu²⁺ to reach the electrode fast enough to keep up; convection is
what keeps the Nernst diffusion layer thin enough for that to happen.

When convection can no longer resupply Cu²⁺ at the demanded rate, the cell
polarizes toward its compliance limit instead of depositing copper — the working
electrode potential leaves its normal pulsing envelope and runs away. This
"collapse" is the central observable: how long a given stirring condition sustains
deposition before mass transport fails.

The quantity of interest is the intra-pulse potential change:

```
ΔE = E(pulse end) − E(pulse start, after switching transient)
```

**Experimental conditions.** 1.57 M CuSO₄·5H₂O with H₂SO₄; cathodic pulse
−500 mA/cm² for 10 ms, 30 ms rest, reverse pulse +100 mA/cm² for 2 ms, 2 ms rest.
Variables under test: stir bar size (none / small / medium), rotation speed
(0–1500 rpm), and bar position (middle / near working electrode).

---

## Installation

```bash
pip install galvani numpy pandas matplotlib
```

`galvani` provides the `.mpr` binary reader. The rest ship with Anaconda.

## Usage

```bash
# process specific files
python plot_deltaE.py "run1.mpr" "run2.mpr"

# or every .mpr next to the script
python plot_deltaE.py
```

Experimental conditions are parsed from EC-Lab filenames, e.g.
`Stirring bar and speed experiment 1 stir 1500rpm middle small_C05.mpr`.
Unrecognized names print a warning and fall back to `unknown` — extend
`STIR_BAR_KEYWORDS` / `POSITION_KEYWORDS` to add naming conventions.

---

## How it works

Four analysis decisions do the real work:

**Transient exclusion by time, not point count.** Immediately after each current
switch there is a fast ohmic jump that isn't part of the electrochemical response.
It's excluded by discarding samples recorded within 1.0 ms (cathodic) or 0.35 ms
(reverse) of pulse start. A time criterion is necessary because EC-Lab's
threshold-based recording sometimes stores only 3–4 unevenly spaced points per
pulse late in a run, so a fixed point offset would cut inconsistently.

**Pause removal.** Gaps between consecutive samples longer than 0.5 s (normal
spacing is ≤ ~30 ms) are treated as experiment pauses and subtracted, producing an
effective time axis so runs interrupted mid-experiment stay comparable.

**Collapse detection.** Ewe is binned into 300 equal-time bins and the min/max
envelope tracked per bin. A baseline envelope is taken as the median over the
first 15% of the run; collapse is the first bin exceeding that envelope by more
than 1 V *and* staying outside for three consecutive bins, so a single noisy bin
doesn't trigger a false positive. Post-collapse pulses are excluded from
steady-state statistics.

**Bad-run flagging.** Real pulse ΔE for this technique is tens of mV. A median
|ΔE| above 500 mV — usually paired with an anomalously low pulse count — indicates
a bad contact or disconnected electrode rather than a real condition, and is
flagged so failed runs don't silently distort cross-condition comparisons.

---

## Outputs

Per file:
- `*_deltaE_vs_t.png` — ΔE vs. effective time for both pulse types; light trace is
  every pulse, bold trace is the binned average, dashed line marks detected collapse
- `*_deltaE_per_pulse.csv` — per-pulse ΔE with original and effective timestamps

Across all files:
- `comparison_deltaE_vs_t.png` — all runs overlaid, colored by bar size, styled by position
- `summary_all_runs.csv` — one row per run: conditions, collapse time, steady-state median ΔE
- `summary_vs_rpm.png` — steady-state ΔE and collapse time vs. rpm

Typical runs contain 160,000–310,000 pulses per sequence.

---

## Findings

Small bar, middle position:

| Stir speed | Time to collapse |
|---|---|
| 0 rpm | 1.52 h |
| 500 rpm | 2.92 h |
| 1000 rpm | 2.77 h |
| 1500 rpm | 2.63 h |

Two results:

**Introducing convection nearly doubles deposition lifetime.** 0 → 500 rpm extends
time-to-collapse from 1.52 h to 2.92 h — the expected mass-transport result.

**More stirring does not help further.** Above 500 rpm, collapse time declines
slightly rather than continuing to improve. Plausible mechanisms at high rpm in a
small-bore cell include vortex formation or turbulent, non-uniform flow rather
than smooth laminar sweep, and mechanical perturbation of the electrode mask.

---

## Limitations

- **n = 1 per condition.** Collapse time is sensitive to contact quality, cell
  filling, and stray bubbles. The shape of the rpm response is not confirmed until
  conditions are repeated.
- **Electrolyte reuse is a confound.** Runs drawing from a shared recycled bath
  sample it at different points in its depletion history. Copper plated out of
  solution does not return when spent electrolyte is poured back, so later runs
  face a measurably weaker bath — which independently produces earlier collapse.
  Repeat runs in a second session collapsed sooner than their originals in the
  same direction, consistent with depletion rather than random variation. Directly
  compared conditions should use a single fresh batch.
- **Speed axis only.** Bar size and position are parsed and plotted but not yet
  varied experimentally.

---

## Context

Written for an undergraduate research project in the Hsu Lab, Pritzker School of
Molecular Engineering, University of Chicago, supporting a graduate study on
edge-growth suppression in Cu-on-Cu electrodeposition.
