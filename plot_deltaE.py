# -*- coding: utf-8 -*-
"""
Per-pulse Delta-E analysis for EC-Lab PRED (pulse reverse electrodeposition) data.
===================================================================================

WHAT THIS SCRIPT DOES
---------------------
Reads BioLogic EC-Lab .mpr files recorded with a PRED technique whose sequences are:
    Ns = 0 : cathodic pulse  (e.g. -500 mA/cm2, 10 ms)
    Ns = 1 : rest            (30 ms)
    Ns = 2 : reverse (anodic) pulse (e.g. +100 mA/cm2, 2 ms)
    Ns = 3 : rest            (2 ms)

For every pulse of Ns=0 and Ns=2 it computes

    Delta E = E(pulse end) - E(start of the electrochemical reaction segment)

i.e. the potential change during the pulse, EXCLUDING the fast voltage jump that
happens right after the current switches direction (the "switching transient").
The transient is removed by a TIME criterion, not by a point count, because
EC-Lab's threshold-based recording sometimes stores only 3-4 unevenly spaced
points per pulse later in a run:
    Ns=0 : use points recorded >= 1.0 ms after the pulse starts
    Ns=2 : use points recorded >= 0.35 ms after the pulse starts

Experiment pauses are removed from the time axis: any gap between two
consecutive data points longer than PAUSE_GAP_S (default 0.5 s; normal gaps are
<= ~30 ms) is treated as a pause and subtracted, giving an "effective time".

NEW IN THIS VERSION (stir bar / speed / position study)
---------------------------------------------------------
Your EC-Lab experiment names now encode the variables under test, e.g.:
    "Stirring bar and speed experiment 1 stir 1500rpm middle small_C05.mpr"
    "Stirring bar and speed experiment 1 stir 1000 rpm middle small_C04.mpr"
instead of the old "..._2ms_2ms_<descriptor>_705F..." naming. `parse_conditions()`
pulls out:
    stir_bar : 'none' | 'small' | 'medium' | 'large' | 'unknown'
    rpm      : int or None
    position : 'middle' | 'close' | 'unknown'
from the file name (case-insensitive, tolerant of "1500rpm" vs "1500 rpm").
CHECK THE PRINTED WARNINGS the first time you run this on a new batch of files -
any 'unknown' means the keyword list below didn't recognize your naming and you
should either rename the file or add a keyword to STIR_BAR_KEYWORDS / POSITION_KEYWORDS.

Because your last run was left going 4 h instead of 2 h, Ewe eventually ran out of
its normal pulsing envelope and pinned against the instrument's compliance/E-range
limit -- i.e. the cell could no longer sustain the programmed current by reducing
Cu2+ alone (mass-transport / reactant depletion at the cathode, exactly what the
stirring study is meant to probe) and the electrode moved into a different regime
(commonly H2 evolution / water breakdown territory). Delta E values recorded after
that point are not representative of real deposition behaviour and will skew any
comparison across stir conditions if left in. `detect_collapse()` flags the time
this happens (looking for the point where the pulse envelope permanently leaves its
initial band) so it can be (a) marked on the per-file plot and (b) excluded from the
steady-state summary statistics used for cross-condition comparison.

OUTPUT (written to outdir)
---------------------------------------
Per file (next to each .mpr, same as before, plus a collapse marker):
  <name>_deltaE_vs_t.png        two panels: Delta E vs effective time for Ns=0 and Ns=2
                                (light trace = every pulse, bold trace = binned average,
                                 dashed vertical line = detected collapse time, if any)
  <name>_deltaE_per_pulse.csv   per-pulse values: original time, effective time, Delta E

Across all files:
  comparison_deltaE_vs_t.png    all processed datasets overlaid (binned averages only),
                                 colored by stir bar size, line style by position
  summary_all_runs.csv          one row per file: parsed conditions, collapse time,
                                 and steady-state (pre-collapse) median Delta E for
                                 both pulse types
  summary_vs_rpm.png            steady-state median dE (Ns=0 and Ns=2) and collapse
                                 time, each vs rpm, colored by stir bar size, line
                                 style by position -- this is the main plot for
                                 answering "how does stirring affect the pulse /
                                 reverse Delta E and how long deposition survives"

HOW TO RUN
----------
1) Install the reader library once:      pip install galvani
   (numpy, pandas and matplotlib are also required - they come with Anaconda)
2) Process specific files:               python plot_deltaE.py "file1.mpr" "file2.mpr"
   or process every .mpr in the folder:  python plot_deltaE.py
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # render to file, no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from galvani import BioLogic

# ----------------------------- settings -------------------------------------
T_SKIP_NS0 = 0.0009    # s; ignore the first ~1 ms of each Ns=0 pulse (transient)
T_SKIP_NS2 = 0.00035   # s; ignore the first ~0.4 ms of each Ns=2 pulse (transient)
PAUSE_GAP_S = 0.5      # s; a recording gap longer than this counts as a pause
NBINS = 4000           # number of time bins for the smoothed trend line
LABEL_NS0 = "Seq 0: $-$500 mA/cm$^2$, 10 ms pulse"
LABEL_NS2 = "Seq 2: +100 mA/cm$^2$, 2 ms pulse"

# collapse (voltage-runaway) detection
COLLAPSE_NBINS = 300        # coarser bins than NBINS - collapse detection just needs the envelope shape
COLLAPSE_BASELINE_FRAC = 0.15   # fraction of the run used to define the "normal" envelope
COLLAPSE_DEV_V = 1.0        # volts outside the normal envelope that counts as departure

# filename keyword lists - EDIT THESE if a file prints as 'unknown'
STIR_BAR_KEYWORDS = {
    "none": [r"\bstir\s*bar\s*none\b", r"\bno\s*bar\b", r"\bwithout\s*bar\b", r"\bbar\s*removed\b"],
    "small": [r"\bsmall\b"],
    "medium": [r"\bmedium\b", r"\bmed\b"],
    "large": [r"\blarge\b"],
}
POSITION_KEYWORDS = {
    "middle": [r"\bmiddle\b", r"\bcenter\b", r"\bcentre\b"],
    "close": [r"\bclose\b", r"\bnear\b", r"\bedge\b"],
}
# -----------------------------------------------------------------------------


def parse_conditions(path):
    """Extract stir bar size, rotation speed (rpm), and electrode position from
    an EC-Lab experiment file name. stir_bar and rpm are independent variables
    (a bar can be present but stationary) so they are parsed independently -
    do not assume rpm == 0 implies no stir bar.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # underscore counts as a "word" character in regex, so \b won't split on it
    # (e.g. "small_C05" would read as one token) - normalize separators to spaces first
    low = re.sub(r"[_\-]+", " ", name.lower())

    m = re.search(r"(\d+)\s*rpm", low)
    if m:
        rpm = int(m.group(1))
    elif re.search(r"\bno\s+stirring\b", low) or re.search(r"\bno\s+stir\b", low):
        rpm = 0  # e.g. "..._no_stirring_..." - bar may still be physically present but stationary
    else:
        rpm = None

    stir_bar = "unknown"
    for label, patterns in STIR_BAR_KEYWORDS.items():
        if any(re.search(p, low) for p in patterns):
            stir_bar = label
            break

    position = "unknown"
    for label, patterns in POSITION_KEYWORDS.items():
        if any(re.search(p, low) for p in patterns):
            position = label
            break

    if stir_bar == "unknown" or position == "unknown" or rpm is None:
        print(f"  [warn] could not fully parse conditions from '{name}' "
              f"-> stir_bar={stir_bar}, rpm={rpm}, position={position} "
              f"(edit STIR_BAR_KEYWORDS / POSITION_KEYWORDS or rename the file)")

    return {"stir_bar": stir_bar, "rpm": rpm, "position": position, "raw": name}


def label_from_conditions(cond):
    rpm_str = f"{cond['rpm']}rpm" if cond["rpm"] is not None else "rpm?"
    return f"{cond['stir_bar']}_{rpm_str}_{cond['position']}"


def remove_pauses(t):
    """Return an 'effective time' axis with experiment pauses subtracted."""
    dt = np.diff(t)
    pause = dt > PAUSE_GAP_S
    for i in np.flatnonzero(pause):
        print(f"  pause detected at t={t[i]:.1f} s, duration {dt[i]:.1f} s")
    excess = np.concatenate(([0.0], np.where(pause, dt, 0.0)))
    return t - np.cumsum(excess)


def segment_dE(ns, t, ewe, target_ns, t_skip, t_eff):
    """Per-pulse Delta E for one sequence number.

    A pulse is a contiguous run of rows with Ns == target_ns.
    Delta E = E(last point) - E(first point recorded >= t_skip after pulse start).
    """
    # boundaries of every contiguous Ns segment
    change = np.flatnonzero(np.diff(ns) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(ns)]))

    # time offset of every sample from the start of its own segment
    offset = t - np.repeat(t[starts], ends - starts)

    mask = ns[starts] == target_ns
    s, e = starts[mask], ends[mask]

    # drop a trailing segment that's cut short by the very end of the recording
    # (e.g. the run was stopped manually mid-pulse). Its 'end' boundary would
    # equal len(t), which np.minimum.reduceat below cannot use as an index, and
    # a truncated pulse isn't a reliable dE measurement anyway.
    if len(e) and e[-1] == len(t):
        s, e = s[:-1], e[:-1]

    if len(s) == 0:
        return pd.DataFrame({"t_s": [], "t_eff_s": [], "dE_V": []})

    # index of the first sample past the transient, found per segment
    idx = np.arange(len(t))
    cand = np.where(offset >= t_skip, idx, len(t))
    bounds = np.column_stack([s, e]).ravel()
    first_ok = np.minimum.reduceat(cand, bounds)[::2]

    valid = first_ok < (e - 1)  # a start point exists and is not the last point
    s, e, first_ok = s[valid], e[valid], first_ok[valid]

    return pd.DataFrame({
        "t_s": t[s],                       # original experiment time of the pulse
        "t_eff_s": t_eff[s],               # time with pauses removed
        "dE_V": ewe[e - 1] - ewe[first_ok],
    })


def bin_average(t, y, nbins=NBINS):
    """Average y into equal-count time bins -> a clean trend line."""
    idx = np.linspace(0, len(t), nbins + 1).astype(int)
    tb = np.array([t[a:b].mean() for a, b in zip(idx[:-1], idx[1:]) if b > a])
    yb = np.array([y[a:b].mean() for a, b in zip(idx[:-1], idx[1:]) if b > a])
    return tb, yb


def detect_collapse(t_eff, ewe, nbins=COLLAPSE_NBINS,
                     baseline_frac=COLLAPSE_BASELINE_FRAC, dev_volts=COLLAPSE_DEV_V):
    """Detect the time at which Ewe permanently leaves its initial pulsing envelope,
    i.e. the point where the cell can no longer sustain the programmed current by
    depositing/stripping copper and instead runs toward the compliance limit.

    Method: bin Ewe into `nbins` equal-time bins and track the min/max (envelope)
    per bin. The "normal" envelope is the median min/max over the first
    `baseline_frac` of the run. The collapse time is the first bin whose envelope
    exceeds the normal one by more than `dev_volts` AND stays out for >= 3
    consecutive bins (so a single noisy bin doesn't trigger a false positive).
    Returns collapse time in effective seconds, or None if no collapse is detected.
    """
    if t_eff[-1] <= t_eff[0]:
        return None
    edges = np.linspace(t_eff[0], t_eff[-1], nbins + 1)
    idx = np.clip(np.digitize(t_eff, edges) - 1, 0, nbins - 1)
    hi = np.full(nbins, np.nan)
    lo = np.full(nbins, np.nan)
    for i in range(nbins):
        sel = ewe[idx == i]
        if sel.size:
            hi[i] = sel.max()
            lo[i] = sel.min()
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_base = max(3, int(nbins * baseline_frac))
    base_hi = np.nanmedian(hi[:n_base])
    base_lo = np.nanmedian(lo[:n_base])

    out = np.flatnonzero((hi > base_hi + dev_volts) | (lo < base_lo - dev_volts))
    if len(out) == 0:
        return None
    out_set = set(out.tolist())
    for i in out:
        if i + 2 < nbins and all((i + k) in out_set for k in (0, 1, 2)):
            return float(centers[i])
    return float(centers[out[0]])


def process(path, outdir):
    """Read one .mpr file, compute per-pulse Delta E, save figure + CSV.
    Returns a summary dict (one row for the cross-run comparison)."""
    cond = parse_conditions(path)
    short = label_from_conditions(cond) + "__" + cond["raw"][-40:].strip("_ ")
    print(f"reading {cond['raw']} ({label_from_conditions(cond)}) ...")
    data = BioLogic.MPRfile(path).data
    ns = data["Ns"]
    t = data["time/s"]
    ewe = data["Ewe/V"].astype(np.float64)

    t_eff = remove_pauses(t)
    collapse_t = detect_collapse(t_eff, ewe)
    if collapse_t is not None:
        print(f"  collapse (voltage runaway) detected at t_eff = {collapse_t/3600:.2f} h "
              f"-- pulses after this are excluded from steady-state stats")
    else:
        print("  no collapse detected in this run")

    df0 = segment_dE(ns, t, ewe, 0, T_SKIP_NS0, t_eff)
    df2 = segment_dE(ns, t, ewe, 2, T_SKIP_NS2, t_eff)
    print(f"  Ns=0: {len(df0)} pulses, median dE = {df0['dE_V'].median()*1000:.2f} mV")
    print(f"  Ns=2: {len(df2)} pulses, median dE = {df2['dE_V'].median()*1000:.2f} mV")

    # sanity check: real pulse dE for this technique is tens of mV. A median in the
    # hundreds of mV or volts (especially paired with a suspiciously low pulse count)
    # usually means a bad contact / disconnected electrode / failed run from the start,
    # not a real electrochemical result - flag it so it doesn't silently distort the
    # cross-condition comparison plots.
    BAD_DE_THRESHOLD_V = 0.5
    for seg_label, dfx in (("Ns=0", df0), ("Ns=2", df2)):
        if len(dfx) and dfx["dE_V"].abs().median() > BAD_DE_THRESHOLD_V:
            print(f"  [warn] {seg_label} median |dE| = {dfx['dE_V'].abs().median()*1000:.0f} mV "
                  f"-- far outside normal pulse behavior (tens of mV). This run likely had a "
                  f"bad contact / disconnected electrode / never established deposition, rather "
                  f"than reflecting a real stirring condition. Consider excluding this file.")

    # steady-state = before the detected collapse (or the whole run if none detected)
    if collapse_t is not None:
        df0_ss = df0[df0["t_eff_s"] < collapse_t]
        df2_ss = df2[df2["t_eff_s"] < collapse_t]
    else:
        df0_ss, df2_ss = df0, df2

    # ---- CSV: one row per pulse (Ns=0 and Ns=2 counts can differ) ----
    n = max(len(df0), len(df2))
    out = pd.DataFrame(index=range(n))
    out["t_s_Ns0"] = pd.Series(df0["t_s"].values)
    out["t_eff_s_Ns0"] = pd.Series(df0["t_eff_s"].values)
    out["dE_Ns0_10ms_V"] = pd.Series(df0["dE_V"].values)
    out["t_s_Ns2"] = pd.Series(df2["t_s"].values)
    out["t_eff_s_Ns2"] = pd.Series(df2["t_eff_s"].values)
    out["dE_Ns2_2ms_V"] = pd.Series(df2["dE_V"].values)
    out.to_csv(os.path.join(outdir, f"{short}_deltaE_per_pulse.csv"),
               index=False, float_format="%.6g")

    # ---- figure: Delta E vs effective time, one panel per sequence ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, df, df_ss, color, title in [(axes[0], df0, df0_ss, "tab:blue", LABEL_NS0),
                                        (axes[1], df2, df2_ss, "tab:red", LABEL_NS2)]:
        th = df["t_eff_s"].values / 3600.0
        y = df["dE_V"].values * 1000.0
        ax.plot(th, y, color=color, alpha=0.15, lw=0.3)   # every pulse
        tb, yb = bin_average(th, y)
        ax.plot(tb, yb, color=color, lw=1.5)              # binned average
        if collapse_t is not None:
            ax.axvline(collapse_t / 3600.0, color="k", ls="--", lw=1,
                       label="collapse detected")
            ax.legend(fontsize=8, loc="upper right")
        # scale to the steady-state (pre-collapse) signal, not the post-collapse
        # runaway - otherwise a handful of extreme post-collapse points stretch the
        # axis so much that the real tens-of-mV pulse behavior looks like a flat line
        y_ref = (df_ss["dE_V"].values * 1000.0) if len(df_ss) else y
        lo, hi = np.percentile(y_ref, [0.5, 99.5])         # clip outlier spikes
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t (h, pauses removed)")
        ax.set_ylabel(r"$\Delta E$ (mV)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{label_from_conditions(cond)}  ({cond['raw']})\n"
                 r"$\Delta E = E_{\rm pulse\,end} - E_{\rm pulse\,start}$"
                 " (switching transition excluded)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(outdir, f"{short}_deltaE_vs_t.png"), dpi=200)
    plt.close(fig)
    print(f"  saved {short}_deltaE_vs_t.png")

    return {
        "file": cond["raw"],
        "stir_bar": cond["stir_bar"],
        "rpm": cond["rpm"],
        "position": cond["position"],
        "collapse_h": None if collapse_t is None else collapse_t / 3600.0,
        "n_pulses_Ns0": len(df0),
        "n_pulses_Ns2": len(df2),
        "median_dE_Ns0_mV_ss": df0_ss["dE_V"].median() * 1000.0 if len(df0_ss) else np.nan,
        "median_dE_Ns2_mV_ss": df2_ss["dE_V"].median() * 1000.0 if len(df2_ss) else np.nan,
        "short_label": short,
    }


def comparison_plot(outdir, summary_rows):
    """Overlay the binned Delta-E trends of every dataset, colored by stir bar
    size and line-styled by position, with rpm shown in the legend."""
    csvs = sorted(glob.glob(os.path.join(outdir, "*_deltaE_per_pulse.csv")))
    if not csvs:
        return
    row_by_short = {r["short_label"]: r for r in summary_rows}

    stir_colors = {"none": "0.5", "small": "tab:green", "medium": "tab:orange",
                   "large": "tab:purple", "unknown": "0.8"}
    pos_styles = {"middle": "-", "close": "--", "unknown": ":"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in csvs:
        short = os.path.basename(path).replace("_deltaE_per_pulse.csv", "")
        row = row_by_short.get(short, {})
        color = stir_colors.get(row.get("stir_bar", "unknown"), "0.8")
        style = pos_styles.get(row.get("position", "unknown"), ":")
        rpm = row.get("rpm")
        label = f"{row.get('stir_bar','?')}, {rpm if rpm is not None else '?'} rpm, {row.get('position','?')}"
        df = pd.read_csv(path)
        for ax, tcol, ycol in [(axes[0], "t_eff_s_Ns0", "dE_Ns0_10ms_V"),
                               (axes[1], "t_eff_s_Ns2", "dE_Ns2_2ms_V")]:
            sub = df[[tcol, ycol]].dropna()
            if sub.empty:
                continue
            tb, yb = bin_average(sub[tcol].values / 3600.0, sub[ycol].values * 1000.0)
            ax.plot(tb, yb, lw=1.5, color=color, ls=style, label=label)
    axes[0].set_title(LABEL_NS0)
    axes[1].set_title(LABEL_NS2)
    for ax in axes:
        ax.set_xlabel("t (h, pauses removed)")
        ax.set_ylabel(r"$\Delta E$ (mV)")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # de-duplicate
    fig.legend(by_label.values(), by_label.keys(), loc="lower center",
               ncol=3, fontsize=8, frameon=False)
    fig.suptitle(r"Intra-pulse $\Delta E = E_{\rm end} - E_{\rm start}$ vs time"
                 "\ncolor = stir bar size, line style = position", fontsize=12)
    fig.tight_layout(rect=[0, 0.18, 1, 0.92])
    fig.savefig(os.path.join(outdir, "comparison_deltaE_vs_t.png"), dpi=200)
    plt.close(fig)
    print("saved comparison_deltaE_vs_t.png")


def summary_plot(outdir, summary_rows):
    """Steady-state median Delta E (Ns=0, Ns=2) and collapse time, each vs rpm,
    colored by stir bar size and line-styled by position. This is the main plot
    for the stir bar / speed / position comparison."""
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(outdir, "summary_all_runs.csv"), index=False, float_format="%.6g")
    print("saved summary_all_runs.csv")

    df_plot = df.dropna(subset=["rpm"])
    if df_plot.empty:
        print("  [warn] no runs with a parsed rpm - skipping summary_vs_rpm.png")
        return

    stir_colors = {"none": "0.5", "small": "tab:green", "medium": "tab:orange",
                   "large": "tab:purple", "unknown": "0.8"}
    pos_markers = {"middle": "o", "close": "s", "unknown": "x"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ycols = ["median_dE_Ns0_mV_ss", "median_dE_Ns2_mV_ss", "collapse_h"]
    titles = [f"Steady-state {LABEL_NS0}", f"Steady-state {LABEL_NS2}",
              "Time to collapse (voltage runaway)"]
    ylabels = [r"median $\Delta E$ (mV)", r"median $\Delta E$ (mV)", "collapse time (h)"]

    for ax, ycol, title, ylabel in zip(axes, ycols, titles, ylabels):
        for (stir_bar, position), grp in df_plot.groupby(["stir_bar", "position"]):
            grp = grp.sort_values("rpm")
            yvals = grp[ycol]
            if ycol == "collapse_h":
                yvals = yvals.fillna(grp["rpm"].map(lambda _: np.nan))  # leave gaps for "no collapse"
            ax.plot(grp["rpm"], yvals,
                     color=stir_colors.get(stir_bar, "0.8"),
                     marker=pos_markers.get(position, "x"),
                     label=f"{stir_bar}, {position}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("stir speed (rpm)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center",
               ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Effect of stir bar size / speed / position on pulse behaviour", fontsize=12)
    fig.tight_layout(rect=[0, 0.14, 1, 0.93])
    fig.savefig(os.path.join(outdir, "summary_vs_rpm.png"), dpi=200)
    plt.close(fig)
    print("saved summary_vs_rpm.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]                       # files given on the command line
        outdir = os.path.dirname(os.path.abspath(files[0]))
    else:
        outdir = os.path.dirname(os.path.abspath(__file__))
        files = sorted(glob.glob(os.path.join(outdir, "*.mpr")))
        if not files:
            sys.exit("No .mpr files found next to this script.")

    summary_rows = [process(f, outdir) for f in files]
    comparison_plot(outdir, summary_rows)
    summary_plot(outdir, summary_rows)