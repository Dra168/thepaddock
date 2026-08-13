"""
The Paddock - qualifying report.

Handles both Qualifying and Sprint Qualifying, which are structurally identical:
three knockout segments, so OpenF1 returns duration and gap_to_leader as
three-element arrays.

    python quali_report.py --dry-run
    python quali_report.py --session "Sprint Qualifying" --year 2025 --round Belgium
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

import common as c

SEGMENTS = ["Q1", "Q2", "Q3"]


def segment_labels(session_name):
    """Sprint qualifying segments are SQ1/SQ2/SQ3 on the broadcast."""
    return ["SQ1", "SQ2", "SQ3"] if "Sprint" in session_name else SEGMENTS


def best_times(results, meta):
    """One row per driver: position, code, colour, best lap, gap to pole,
    and the segment they were knocked out in."""
    rows = []
    for _, r in results.iterrows():
        num = r.get("driver_number")
        if num is None or int(num) not in meta:
            continue
        info = meta[int(num)]
        times = [c.seg_value(r.get("duration"), i) for i in range(3)]
        reached = sum(1 for t in times if t is not None)
        best = min((t for t in times if t is not None), default=None)
        rows.append({
            "position": r.get("position"),
            "code": info["code"],
            "team": info["team"],
            "color": info["color"],
            "linestyle": info.get("linestyle", "-"),
            "q1": times[0], "q2": times[1], "q3": times[2],
            "best": best,
            "reached": reached,
            "dsq": bool(r.get("dsq")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pole = df["best"].min()
    df["gap"] = df["best"] - pole
    return df.sort_values("position", na_position="last").reset_index(drop=True)


def chart_gap_to_pole(df, labels, title):
    def draw(path):
        data = df.dropna(subset=["gap"])
        if data.empty:
            raise ValueError("no lap times to plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        c.style_axes(ax)
        ax.barh(data["code"], data["gap"], color=data["color"],
                edgecolor=c.BG, linewidth=0.8)
        for y, (gap, best) in enumerate(zip(data["gap"], data["best"])):
            label = c.fmt_time(best) if y == 0 else f"+{gap:.3f}"
            ax.text(gap + data["gap"].max() * 0.015, y, label,
                    va="center", color=c.FG, fontsize=8)
        ax.set_xlim(0, data["gap"].max() * 1.18)
        ax.invert_yaxis()
        ax.set_xlabel("Gap to pole (s)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.15, color="#666666")
        c.save(fig, path)
    return draw


def chart_segments(df, labels, title):
    """Each driver's time in every segment they reached, as gap to that
    segment's fastest. Shows who found time when it mattered."""
    def draw(path):
        cols = ["q1", "q2", "q3"]
        fig, ax = plt.subplots(figsize=(10, 6))
        c.style_axes(ax)

        plotted = 0
        for _, row in df.iterrows():
            xs, ys = [], []
            for i, col in enumerate(cols):
                t = row[col]
                if t is None or pd.isna(t):
                    continue
                fastest = df[col].min()
                if pd.isna(fastest):
                    continue
                xs.append(i)
                ys.append(t - fastest)
            if len(xs) < 2:
                continue
            ax.plot(xs, ys, marker="o", markersize=4,
                    color=row["color"], linestyle=row.get("linestyle", "-"),
                    linewidth=1.5, label=row["code"])
            plotted += 1

        if plotted == 0:
            raise ValueError("not enough segment times to plot")

        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Gap to segment best (s)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(alpha=0.15, color="#666666")
        ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="small",
                  facecolor="#2a2a2a", labelcolor=c.FG, ncol=1)
        c.save(fig, path)
    return draw


def build_caption(df, labels, sess):
    name = sess.get("session_name", "Qualifying")
    where = sess.get("country_name") or sess.get("circuit_short_name") or ""
    lines = [f"## {where} {sess.get('year', '')} - {name}", ""]

    top = df.dropna(subset=["position"]).head(3)
    for medal, (_, row) in zip(["🥇", "🥈", "🥉"], top.iterrows()):
        gap = "" if row["gap"] == 0 else f" ({c.fmt_gap(row['gap'])})"
        lines.append(f"{medal} **{row['code']}** ({row['team']}) "
                     f"{c.fmt_time(row['best'])}{gap}")
    lines.append("")

    # margin between pole and P2
    if len(df) >= 2 and pd.notna(df.iloc[1]["gap"]):
        lines.append(f"**Pole margin** {df.iloc[1]['gap']:.3f}s")

    # who went out where
    for seg, reached in ((labels[0], 1), (labels[1], 2)):
        out = df[df["reached"] == reached]["code"].tolist()
        if out:
            lines.append(f"**Out in {seg}** {', '.join(out)}")

    # closest call: smallest gap between adjacent cars
    ranked = df.dropna(subset=["best"]).sort_values("best").reset_index(drop=True)
    if len(ranked) >= 2:
        diffs = ranked["best"].diff().iloc[1:]
        if not diffs.empty and pd.notna(diffs.min()):
            i = diffs.idxmin()
            lines.append(f"**Closest** {ranked.loc[i - 1, 'code']} to "
                         f"{ranked.loc[i, 'code']}, {diffs.min():.3f}s")

    lines.append("")
    lines.append("Data via OpenF1. Argue about it below.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="Qualifying",
                        choices=["Qualifying", "Sprint Qualifying"])
    c.add_common_args(parser, "Qualifying")
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

    df = best_times(results, meta)
    if df.empty:
        print("no usable lap times")
        return 0

    labels = segment_labels(args.session)
    where = sess.get("country_name") or ""
    year = sess.get("year", "")
    prefix = f"{where} {year} - {args.session}"

    made = c.build_charts([
        ("quali_gap.png", chart_gap_to_pole(df, labels, f"{prefix}: gap to pole")),
        ("quali_segments.png", chart_segments(df, labels, f"{prefix}: segment by segment")),
    ])
    return c.post_to_discord(build_caption(df, labels, sess), made, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
