import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd
import common as c

PRACTICE_NAMES = ["Practice 1", "Practice 2", "Practice 3"]

MIN_RUN_LAPS = 4
QUICKLAP_THRESHOLD = 1.10  

def short_name(session_name):
    digits = "".join(ch for ch in str(session_name) if ch.isdigit())
    return f"FP{digits}" if digits else str(session_name)

def best_laps(results, meta):
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
            "best": c.scalar_value(r.get("duration")),
            "laps": r.get("number_of_laps"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["gap"] = df["best"] - df["best"].min()
    return df.sort_values("position", na_position="last").reset_index(drop=True)

def long_run_frame(laps, stints, meta):
    if laps.empty:
        return pd.DataFrame()

    df = laps.dropna(subset=["seconds"]).copy()
    if "is_pit_out_lap" in df.columns:
        df = df[df["is_pit_out_lap"] != True]  
    if df.empty:
        return pd.DataFrame()

    df["compound"] = "UNKNOWN"
    df["run"] = 0
    if not stints.empty:
        for _, st in stints.iterrows():
            num, start = st.get("driver_number"), st.get("lap_start")
            if pd.isna(start):
                continue
            end = st.get("lap_end")
            end = end if pd.notna(end) else df["lap_number"].max()
            sel = ((df["driver_number"] == num)
                   & (df["lap_number"] >= start) & (df["lap_number"] <= end))
            df.loc[sel, "compound"] = st["compound"]
            df.loc[sel, "run"] = st.get("stint_number", 0)

    keep = [g for _, g in df.groupby(["driver_number", "run"]) if len(g) >= MIN_RUN_LAPS]
    if not keep:
        return pd.DataFrame()

    out = pd.concat(keep, ignore_index=True)
    out = out[out["seconds"] <= out["seconds"].min() * QUICKLAP_THRESHOLD]
    out["code"] = out["driver_number"].map(
        lambda n: meta[int(n)]["code"] if int(n) in meta else str(n))
    return out

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


def chart_long_runs(runs, meta, title, n=10):
    def draw(path):
        if runs.empty:
            raise ValueError("no long runs found")
        medians = runs.groupby("driver_number")["seconds"].median().sort_values()
        data, colors, labels = [], [], []
        for num in list(medians.index[:n]):
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


def build_caption(blocks, meeting, long_run_session):
    where = meeting.get("country_name") or meeting.get("circuit_short_name") or ""
    year = meeting.get("year", "")
    names = " and ".join(b["short"] for b in blocks)
    lines = [f"## {where} {year} - {names}", ""]

    for block in blocks:
        df = block["df"]
        lines.append(f"**{block['short']}**")
        for i, (_, row) in enumerate(df.dropna(subset=["best"]).head(3).iterrows()):
            gap = "" if row["gap"] == 0 else f" ({c.fmt_gap(row['gap'])})"
            lines.append(f"{i + 1}. {row['code']} ({row['team']}) "
                         f"{c.fmt_time(row['best'])}{gap}")
        lines.append("")

    if long_run_session is not None:
        runs = long_run_session["runs"]
        medians = runs.groupby("code")["seconds"].median().sort_values()
        if len(medians):
            leader = medians.iloc[0]
            chasers = ", ".join(
                f"{code} (+{val - leader:.3f}s)"
                for code, val in medians.iloc[1:3].items()
            )
            line = (f"**Long-run pace ({long_run_session['short']})** "
                    f"{medians.index[0]} {c.fmt_time(leader)} median")
            if chasers:
                line += f", then {chasers}"
            lines.append(line)
            used = sorted(set(runs["compound"]) - {"UNKNOWN"})
            if used:
                lines.append(f"**Compounds run** {', '.join(t.title() for t in used)}")
    else:
        lines.append("_No long runs long enough to analyse._")

    busiest = pd.concat([b["df"] for b in blocks]).dropna(subset=["laps"])
    if not busiest.empty:
        totals = busiest.groupby("code")["laps"].sum().sort_values(ascending=False)
        lines.append(f"**Most laps** {totals.index[0]} ({int(totals.iloc[0])})")

    lines.append("")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    c.add_common_args(parser, "Practice")
    parser.set_defaults(lookback=22)
    args = parser.parse_args()

    if args.year:
        sessions = c.find_sessions(args.year, PRACTICE_NAMES, args.round)
        if not sessions:
            print(f"no practice sessions found for {args.year} {args.round or ''}")
            return 0
    else:
        sessions = c.find_recent_sessions(PRACTICE_NAMES, args.lookback)
        if not sessions:
            print(f"no practice session finished in the last {args.lookback}h, "
                  f"nothing to do")
            return 0

    print(f"found {len(sessions)}: "
          f"{', '.join(short_name(s.get('session_name')) for s in sessions)}")

    blocks = []
    for sess in sessions:
        key = sess["session_key"]
        meta = c.load_drivers(key)
        results = c.session_results(key)
        if results.empty or not meta:
            print(f"  {short_name(sess.get('session_name'))}: no results yet, skipping")
            continue
        df = best_laps(results, meta)
        if df.empty:
            print(f"  {short_name(sess.get('session_name'))}: no usable times, skipping")
            continue
        runs = long_run_frame(c.session_laps(key), c.session_stints(key), meta)
        blocks.append({"sess": sess, "short": short_name(sess.get("session_name")),
                       "df": df, "runs": runs, "meta": meta})
        print(f"  {short_name(sess.get('session_name'))}: "
              f"{len(df)} drivers, {len(runs)} long-run laps")

    if not blocks:
        print("nothing usable in any session")
        return 0

    meeting = blocks[-1]["sess"]
    where = meeting.get("country_name") or ""
    year = meeting.get("year", "")

    with_runs = [b for b in blocks if not b["runs"].empty]
    long_run_session = max(with_runs, key=lambda b: len(b["runs"])) if with_runs else None

    specs = [
        (f"practice_{b['short'].lower()}_timesheet.png",
         chart_timesheet(b["df"], f"{where} {year} - {b['short']} timesheet"))
        for b in blocks
    ]
    if long_run_session is not None:
        specs.append((
            "practice_long_runs.png",
            chart_long_runs(long_run_session["runs"], long_run_session["meta"],
                            f"{where} {year} - {long_run_session['short']} long-run pace"),
        ))

    made = c.build_charts(specs)
    caption = build_caption(blocks, meeting, long_run_session)
    return c.post_to_discord(caption, made, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
