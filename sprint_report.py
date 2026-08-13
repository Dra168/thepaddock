"""
The Paddock - sprint race report.

Same shape as the Grand Prix report, but built entirely on OpenF1 rather than
Jolpica, because sprint sessions are easier to reach there and the driver
colours come free.

Sprints are short and usually one-stint, so there is no strategy chart. What
matters is the start, the pace order, and who actually gained anything.

    python sprint_report.py --dry-run
    python sprint_report.py --year 2025 --round Belgium
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

import common as c

QUICKLAP_THRESHOLD = 1.07


def classify(results, meta, grid):
    rows = []
    for _, r in results.iterrows():
        num = r.get("driver_number")
        if num is None or int(num) not in meta:
            continue
        info = meta[int(num)]
        rows.append({
            "driver_number": int(num),
            "position": r.get("position"),
            "code": info["code"],
            "team": info["team"],
            "color": info["color"],
            "linestyle": info.get("linestyle", "-"),
            "laps": r.get("number_of_laps"),
            "gap": c.scalar_value(r.get("gap_to_leader")),
            "dnf": bool(r.get("dnf")),
            "grid": grid.get(int(num)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("position", na_position="last").reset_index(drop=True)


def chart_pace(laps, df, meta, title, n=10):
    def draw(path):
        if laps.empty:
            raise ValueError("no lap data available")
        clean = laps.dropna(subset=["seconds"]).copy()
        if "is_pit_out_lap" in clean.columns:
            clean = clean[clean["is_pit_out_lap"] != True]  # noqa: E712
        if clean.empty:
            raise ValueError("no clean laps")
        cutoff = clean["seconds"].min() * QUICKLAP_THRESHOLD
        clean = clean[clean["seconds"] <= cutoff]

        order = df.head(n)["driver_number"].tolist()
        data, colors, labels = [], [], []
        for num in order:
            vals = clean.loc[clean["driver_number"] == num, "seconds"]
            if len(vals) < 3:
                continue
            data.append(vals.values)
            colors.append(meta[num]["color"])
            labels.append(meta[num]["code"])
        if not data:
            raise ValueError("not enough clean laps to plot")

        fig, ax = plt.subplots(figsize=(11, 6))
        c.style_axes(ax)
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        except TypeError:
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


def chart_grid_vs_finish(df, title):
    """A sprint is decided at the start, so plot grid slot against finish."""
    def draw(path):
        data = df.dropna(subset=["position", "grid"])
        if data.empty:
            raise ValueError("no starting grid data available")

        fig, ax = plt.subplots(figsize=(9, 7))
        c.style_axes(ax)
        for _, row in data.iterrows():
            ax.plot([0, 1], [row["grid"], row["position"]],
                    color=row["color"], linestyle=row["linestyle"],
                    linewidth=1.8, marker="o", markersize=5)
            ax.text(-0.04, row["grid"], row["code"], ha="right", va="center",
                    color=c.FG, fontsize=8)
            ax.text(1.04, row["position"], row["code"], ha="left", va="center",
                    color=c.FG, fontsize=8)

        ax.set_xlim(-0.25, 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Grid", "Finish"])
        ax.invert_yaxis()
        ax.set_ylabel("Position")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15, color="#666666")
        c.save(fig, path)
    return draw


def build_caption(df, laps, meta, sess):
    where = sess.get("country_name") or sess.get("circuit_short_name") or ""
    lines = [f"## {where} {sess.get('year', '')} - Sprint", ""]

    for medal, (_, row) in zip(["🥇", "🥈", "🥉"], df.head(3).iterrows()):
        gap = "" if not row["gap"] else f" ({c.fmt_gap(row['gap'])})"
        lines.append(f"{medal} **{row['code']}** ({row['team']}){gap}")
    lines.append("")

    if not laps.empty:
        clean = laps.dropna(subset=["seconds"])
        if not clean.empty:
            fl = clean.loc[clean["seconds"].idxmin()]
            num = int(fl["driver_number"])
            code = meta.get(num, {}).get("code", str(num))
            lines.append(f"**Fastest lap** {code} {c.fmt_time(fl['seconds'])} "
                         f"on lap {int(fl['lap_number'])}")

    gained = df.dropna(subset=["position", "grid"]).copy()
    if not gained.empty:
        gained["gained"] = gained["grid"] - gained["position"]
        best = gained.sort_values("gained", ascending=False).iloc[0]
        if best["gained"] > 0:
            lines.append(f"**Biggest gain** {best['code']} "
                         f"P{int(best['grid'])} to P{int(best['position'])} "
                         f"(+{int(best['gained'])})")

    out = df[df["dnf"]]["code"].tolist()
    if out:
        lines.append(f"**Did not finish** {', '.join(out)}")

    lines.append("")
    lines.append("Data via OpenF1. Argue about it below.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    c.add_common_args(parser, "Sprint")
    args = parser.parse_args()

    sess = c.resolve_session("Sprint", args)
    if not sess:
        return 0

    key = sess["session_key"]
    meta = c.load_drivers(key)
    results = c.session_results(key)
    if results.empty or not meta:
        print("no results or driver data available yet")
        return 0

    grid = {int(g["driver_number"]): g.get("position")
            for g in c.openf1("starting_grid", session_key=key)
            if g.get("driver_number") is not None}
    if not grid:
        print("no starting grid available; the grid chart will be skipped")

    df = classify(results, meta, grid)
    laps = c.session_laps(key)
    print(f"{len(df)} classified, {len(laps)} lap records, {len(grid)} grid slots")

    where = sess.get("country_name") or ""
    prefix = f"{where} {sess.get('year', '')} - Sprint"

    made = c.build_charts([
        ("sprint_grid.png", chart_grid_vs_finish(df, f"{prefix}: grid to flag")),
        ("sprint_pace.png", chart_pace(laps, df, meta, f"{prefix}: race pace (top 10)")),
    ])
    return c.post_to_discord(build_caption(df, laps, meta, sess), made, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
