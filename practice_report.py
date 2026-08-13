"""
The Paddock - practice report.

Practice is not in Jolpica at all, so everything here comes from OpenF1.

Practice is about two different things at once: one-lap pace on low fuel and
long-run pace on high fuel. The charts separate them, because the headline
timesheet on its own is close to meaningless.

    python practice_report.py --dry-run
    python practice_report.py --session "Practice 2" --year 2025 --round Hungary
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

import common as c

# A long run is a sequence of laps on the same tyre with no pit-out in between.
MIN_RUN_LAPS = 4
QUICKLAP_THRESHOLD = 1.10  # practice fuel loads vary far more than a race


def best_laps(results, meta):
    rows = []
    for _, r in results.iterrows():
        num = r.get("driver_number")
        if num is None or int(num) not in meta:
            continue
        info = meta[int(num)]
        best = c.scalar_value(r.get("duration"))
        rows.append({
            "driver_number": int(num),
            "position": r.get("position"),
            "code": info["code"],
            "team": info["team"],
            "color": info["color"],
            "best": best,
            "laps": r.get("number_of_laps"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    fastest = df["best"].min()
    df["gap"] = df["best"] - fastest
    return df.sort_values("position", na_position="last").reset_index(drop=True)


def chart_timesheet(df, title):
    def draw(path):
        data = df.dropna(subset=["gap"])
        if data.empty:
            raise ValueError("no lap times to plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        c.style_axes(ax)
        ax.barh(data["code"], data["gap"], color=data["color"],
                edgecolor=c.BG, linewidth=0.8)
        span = max(data["gap"].max(), 0.001)
        for y, (gap, best, laps) in enumerate(
                zip(data["gap"], data["best"], data["laps"])):
            lap_txt = f"  {int(laps)} laps" if pd.notna(laps) else ""
            label = c.fmt_time(best) if y == 0 else f"+{gap:.3f}"
            ax.text(gap + span * 0.015, y, label + lap_txt,
                    va="center", color=c.FG, fontsize=8)
        ax.set_xlim(0, span * 1.25)
        ax.invert_yaxis()
        ax.set_xlabel("Gap to fastest (s)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.15, color="#666666")
        c.save(fig, path)
    return draw


def long_run_frame(laps, stints, meta):
    """Laps that belong to a run of at least MIN_RUN_LAPS on one tyre.

    Pit-out laps are dropped, then anything outside the threshold, which strips
    out aborted laps, traffic and in-laps without needing fuel-load data.
    """
    if laps.empty:
        return pd.DataFrame()

    df = laps.dropna(subset=["seconds"]).copy()
    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] != True]  # noqa: E712 (may be object dtype)
    if df.empty:
        return pd.DataFrame()

    # tag each lap with the stint it belongs to, so compounds can colour it
    df["compound"] = "UNKNOWN"
    df["run"] = 0
    if not stints.empty:
        for _, st in stints.iterrows():
            num, start, end = st.get("driver_number"), st.get("lap_start"), st.get("lap_end")
            if pd.isna(start):
                continue
            end = end if pd.notna(end) else df["lap_number"].max()
            sel = ((df["driver_number"] == num)
                   & (df["lap_number"] >= start) & (df["lap_number"] <= end))
            df.loc[sel, "compound"] = st["compound"]
            df.loc[sel, "run"] = st.get("stint_number", 0)

    keep = []
    for (num, run), group in df.groupby(["driver_number", "run"]):
        if len(group) >= MIN_RUN_LAPS:
            keep.append(group)
    if not keep:
        return pd.DataFrame()

    out = pd.concat(keep, ignore_index=True)
    cutoff = out["seconds"].min() * QUICKLAP_THRESHOLD
    out = out[out["seconds"] <= cutoff]
    out["code"] = out["driver_number"].map(
        lambda n: meta[int(n)]["code"] if int(n) in meta else str(n))
    return out


def chart_long_runs(runs, meta, title, n=10):
    def draw(path):
        if runs.empty:
            raise ValueError("no long runs found in this session")
        medians = runs.groupby("driver_number")["seconds"].median().sort_values()
        top = list(medians.index[:n])

        data, colors, labels = [], [], []
        for num in top:
            vals = runs.loc[runs["driver_number"] == num, "seconds"]
            if len(vals) < MIN_RUN_LAPS:
                continue
            data.append(vals.values)
            info = meta.get(int(num), {})
            colors.append(info.get("color", c.GREY))
            labels.append(info.get("code", str(num)))
        if not data:
            raise ValueError("not enough long-run laps to plot")

        fig, ax = plt.subplots(figsize=(11, 6))
        c.style_axes(ax)
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        except TypeError:  # matplotlib < 3.9
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for part in ("whiskers", "caps"):
            for item in bp[part]:
                item.set_color("#999999")
        for median in bp["medians"]:
            median.set_color("#ffffff")
            median.set_linewidth(1.6)

        ax.set_ylabel("Lap time (s)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, color="#666666")
        c.save(fig, path)
    return draw


def build_caption(df, runs, sess):
    name = sess.get("session_name", "Practice")
    where = sess.get("country_name") or sess.get("circuit_short_name") or ""
    lines = [f"## {where} {sess.get('year', '')} - {name}", ""]

    top = df.dropna(subset=["best"]).head(3)
    for medal, (_, row) in zip(["🥇", "🥈", "🥉"], top.iterrows()):
        gap = "" if row["gap"] == 0 else f" ({c.fmt_gap(row['gap'])})"
        lines.append(f"{medal} **{row['code']}** ({row['team']}) "
                     f"{c.fmt_time(row['best'])}{gap}")
    lines.append("")

    if not runs.empty:
        medians = runs.groupby("code")["seconds"].median().sort_values()
        if len(medians) >= 1:
            leader = medians.index[0]
            lines.append(f"**Best long-run pace** {leader} "
                         f"({c.fmt_time(medians.iloc[0])} median)")
        if len(medians) >= 2:
            lines.append(f"**Next best** {medians.index[1]} "
                         f"(+{medians.iloc[1] - medians.iloc[0]:.3f}s)")
        used = sorted(set(runs["compound"]) - {"UNKNOWN"})
        if used:
            lines.append(f"**Compounds run** {', '.join(t.title() for t in used)}")
    else:
        lines.append("_No long runs long enough to analyse in this session._")

    busiest = df.dropna(subset=["laps"])
    if not busiest.empty:
        row = busiest.loc[busiest["laps"].idxmax()]
        lines.append(f"**Most laps** {row['code']} ({int(row['laps'])})")

    lines.append("")
    lines.append("One-lap pace is the timesheet; long runs are the real story. "
                 "Data via OpenF1.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="Practice 2",
                        choices=["Practice 1", "Practice 2", "Practice 3"])
    c.add_common_args(parser, "Practice")
    args = parser.parse_args()

    sess = c.resolve_session(args.session, args)
    if not sess:
        return 0

    key = sess["session_key"]
    meta = c.load_drivers(key)
    results = c.session_results(key)
    if results.empty or not meta:
        print("no results or driver data available yet")
        return 0

    df = best_laps(results, meta)
    if df.empty:
        print("no usable lap times")
        return 0

    runs = long_run_frame(c.session_laps(key), c.session_stints(key), meta)
    print(f"{len(df)} drivers, {len(runs)} long-run laps")

    where = sess.get("country_name") or ""
    prefix = f"{where} {sess.get('year', '')} - {args.session}"

    made = c.build_charts([
        ("practice_timesheet.png", chart_timesheet(df, f"{prefix}: timesheet")),
        ("practice_long_runs.png", chart_long_runs(runs, meta, f"{prefix}: long-run pace")),
    ])
    return c.post_to_discord(build_caption(df, runs, sess), made, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
