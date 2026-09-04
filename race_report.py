import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a CI runner
import matplotlib.pyplot as plt
import pandas as pd
import requests

from fastf1 import Cache
from fastf1.ergast import Ergast

CACHE_DIR = Path("cache")
OUT_DIR = Path("out")
LOOKBACK_HOURS = 96
WEBHOOK_USERNAME = "The Paddock"
PAGE_SIZE = 1000         
QUICKLAP_THRESHOLD = 1.07  

TEAM_COLORS = {
    "red_bull": "#3671C6",
    "ferrari": "#E8002D",
    "mercedes": "#27F4D2",
    "mclaren": "#FF8000",
    "aston_martin": "#229971",
    "alpine": "#FF87BC",
    "williams": "#64C4FF",
    "rb": "#6692FF",
    "racing_bulls": "#6692FF",
    "alphatauri": "#6692FF",
    "sauber": "#52E252",
    "kick_sauber": "#52E252",
    "audi": "#52E252",
    "alfa": "#52E252",
    "haas": "#B6BABD",
    "cadillac": "#B08D57",
}
FALLBACK_COLORS = [
    "#8E7CC3", "#D9A441", "#5FA8D3", "#C36B6B",
    "#7FB069", "#B5838D", "#9C89B8", "#E0A458",
]
STINT_SHADES = ["#E8002D", "#F5A623", "#4A90D9", "#7FB069", "#B5838D", "#9C89B8"]

OPENF1_BASE = "https://api.openf1.org/v1"
OPENF1_TIMEOUT = 20
COMPOUND_COLORS = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "UNKNOWN": "#888888",
}

def style_axes(ax):
    ax.set_facecolor("#1e1e1e")
    ax.figure.set_facecolor("#1e1e1e")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors="#dddddd")
    ax.xaxis.label.set_color("#dddddd")
    ax.yaxis.label.set_color("#dddddd")
    ax.title.set_color("#ffffff")


def fetch_all_pages(fn, **kwargs):
    """Call an Ergast/Jolpica endpoint and concatenate every page of results.

    A race has roughly 20 drivers x 60+ laps of lap times, well past the
    per-request cap, so lap times always need more than one page.
    """
    resp = fn(limit=PAGE_SIZE, **kwargs)
    frames = list(resp.content)

    for _ in range(20):
        try:
            resp = resp.get_next_result_page()
        except ValueError:
            break
        frames.extend(resp.content)
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def team_color(constructor_id, index=0):
    if constructor_id in TEAM_COLORS:
        return TEAM_COLORS[constructor_id]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]

def find_recent_race(ergast, lookback_hours=LOOKBACK_HOURS):
    """Return (year, round) for the most recent race that has finished."""
    now = datetime.now(timezone.utc)
    found = None

    for year in (now.year, now.year - 1):
        try:
            schedule = ergast.get_race_schedule(season=year, limit=100)
        except Exception as exc:
            print(f"could not load {year} schedule: {exc}")
            continue

        for _, race in schedule.iterrows():
            date, time = race.get("raceDate"), race.get("raceTime")
            if pd.isna(date):
                continue
            when = pd.Timestamp(date)
            if not pd.isna(time):
                when = pd.Timestamp(f"{pd.Timestamp(date).date()} {time}")
            when = when.tz_localize("UTC") if when.tzinfo is None else when

            age = now - when
            if timedelta(hours=3) < age < timedelta(hours=lookback_hours):
                found = (year, int(race["round"]), race["raceName"])

        if found:
            break

    return found


def fetch_race_meta(ergast, year, rnd):
    try:
        sched = ergast.get_race_schedule(season=year, round=rnd, limit=100)
        if len(sched):
            row = sched.iloc[0]
            return row.get("raceName"), row.get("raceDate")
    except Exception as exc:
        print(f"could not load race metadata: {exc}")
    return None, None


def fetch_openf1_stints(year, race_date, number_to_driver):
    if race_date is None or pd.isna(race_date):
        print("openf1: no race date to match on, skipping compounds")
        return None
    target = pd.Timestamp(race_date).date()

    try:
        sessions = requests.get(
            f"{OPENF1_BASE}/sessions",
            params={"year": int(year), "session_name": "Race"},
            timeout=OPENF1_TIMEOUT,
        )
        sessions.raise_for_status()
        sessions = sessions.json()
    except Exception as exc:
        print(f"openf1: session lookup failed ({exc}), falling back to stint numbers")
        return None

    session_key = None
    for sess in sessions:
        start = sess.get("date_start")
        if not start:
            continue
        try:
            if pd.Timestamp(start).date() == target:
                session_key = sess.get("session_key")
                break
        except Exception:
            continue

    if session_key is None:
        print(f"openf1: no Race session found for {target}, falling back")
        return None

    try:
        resp = requests.get(
            f"{OPENF1_BASE}/stints",
            params={"session_key": session_key},
            timeout=OPENF1_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        print(f"openf1: stint fetch failed ({exc}), falling back to stint numbers")
        return None

    if not raw:
        print("openf1: session had no stint data, falling back")
        return None

    stints = pd.DataFrame(raw)
    stints["driverId"] = stints["driver_number"].map(number_to_driver)
    stints = stints.dropna(subset=["driverId"])
    if stints.empty:
        print("openf1: could not match any driver numbers, falling back")
        return None

    stints["compound"] = (
        stints.get("compound", pd.Series(dtype=str))
        .fillna("UNKNOWN").astype(str).str.upper()
    )
    before = len(stints)
    stints = stints.drop_duplicates(subset=["driverId", "stint_number"], keep="first")
    if len(stints) != before:
        print(f"openf1: dropped {before - len(stints)} duplicate stint record(s)")

    counts = stints.groupby("driverId").size()
    print(f"openf1: matched session {session_key}, {len(stints)} stints "
          f"across {len(counts)} drivers "
          f"(min {counts.min()}, max {counts.max()} per driver)")
    if counts.max() > 5:
        worst = counts.idxmax()
        print(f"openf1: WARNING {worst} has {counts.max()} stints, which is "
              f"implausible; the driver number mapping is probably wrong")

    return stints.sort_values(["driverId", "stint_number"])


def load_race(ergast, year, rnd):
    results = fetch_all_pages(ergast.get_race_results, season=year, round=rnd)
    if results.empty:
        raise ValueError(f"no results available for {year} round {rnd}")

    laps = fetch_all_pages(ergast.get_lap_times, season=year, round=rnd)
    pits = fetch_all_pages(ergast.get_pit_stops, season=year, round=rnd)

    results = results.sort_values("position").reset_index(drop=True)

    meta = {}
    seen_team = {}
    for i, row in results.iterrows():
        code = row.get("driverCode")
        if not isinstance(code, str) or not code:
            code = str(row.get("familyName", row["driverId"]))[:3].upper()
        # teammates share a colour, so vary the linestyle to tell them apart
        cid = row.get("constructorId")
        nth = seen_team.get(cid, 0)
        seen_team[cid] = nth + 1
        meta[row["driverId"]] = {
            "code": code,
            "team": row.get("constructorName", ""),
            "color": team_color(cid, i),
            "linestyle": ["-", "--", ":"][nth % 3],
            "position": row.get("position"),
        }

    if not laps.empty:
        laps = laps.copy()
        laps["code"] = laps["driverId"].map(lambda d: meta.get(d, {}).get("code", d))
        laps["seconds"] = pd.to_timedelta(laps["time"], errors="coerce").dt.total_seconds()

    number_to_driver = {}
    for _, row in results.iterrows():
        val = row.get("number")
        if pd.isna(val):
            val = row.get("driverNumber")
        if pd.notna(val):
            number_to_driver[int(val)] = row["driverId"]

    race_name, race_date = fetch_race_meta(ergast, year, rnd)
    stints = fetch_openf1_stints(year, race_date, number_to_driver)

    return results, laps, pits, stints, meta, race_name

def chart_stints(results, laps, pits, stints, meta, path, title=""):
    """Stint lengths per driver.

    Coloured by tyre compound when OpenF1 supplied them, otherwise by stint
    number using pit stop laps from Jolpica.
    """
    use_compounds = stints is not None and not stints.empty
    if not use_compounds and pits.empty:
        raise ValueError("neither compound nor pit stop data available")

    total = results["laps"].max()
    if pd.isna(total):
        raise ValueError("no lap counts in results")
    total_laps = int(total)
    order = [row["driverId"] for _, row in results.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 10))
    style_axes(ax)
    seen_compounds = []

    for drv in order:
        code = meta[drv]["code"]
        finished = results.loc[results["driverId"] == drv, "laps"]
        last_lap = int(finished.iloc[0]) if len(finished) else total_laps

        if use_compounds:
            rows = stints[stints["driverId"] == drv]
            if rows.empty:
                print(f"openf1: no stint records for {code}, leaving row blank")
                continue
            for _, st in rows.iterrows():
                start = st.get("lap_start")
                if pd.isna(start):
                    continue  
                start = int(start)
                end = st.get("lap_end")
                end = int(end) if pd.notna(end) else last_lap
                width = end - start + 1
                if width <= 0:
                    continue
                comp = st["compound"]
                if comp not in seen_compounds:
                    seen_compounds.append(comp)
                ax.barh(
                    y=code, width=width, left=start - 1,
                    color=COMPOUND_COLORS.get(comp, COMPOUND_COLORS["UNKNOWN"]),
                    edgecolor="#1e1e1e", linewidth=0.8,
                )
        else:
            stop_laps = sorted(pits.loc[pits["driverId"] == drv, "lap"].astype(int).tolist())
            bounds = [0] + stop_laps + [last_lap]
            for i in range(len(bounds) - 1):
                width = bounds[i + 1] - bounds[i]
                if width <= 0:
                    continue
                ax.barh(
                    y=code, width=width, left=bounds[i],
                    color=STINT_SHADES[i % len(STINT_SHADES)],
                    edgecolor="#1e1e1e", linewidth=0.8,
                )

    ax.set_title(title)
    ax.set_xlabel("Lap")
    ax.set_xlim(0, total_laps)
    ax.invert_yaxis()
    ax.grid(False)

    if use_compounds:
        keys = [c for c in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
                if c in seen_compounds]
        labels = [c.title() for c in keys]
        colors = [COMPOUND_COLORS[c] for c in keys]
    else:
        labels = [f"Stint {i + 1}" for i in range(3)]
        colors = [STINT_SHADES[i % len(STINT_SHADES)] for i in range(3)]

    if labels:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
        ax.legend(handles, labels, loc="lower right", facecolor="#2a2a2a",
                  labelcolor="#dddddd", fontsize="small")

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

def chart_race_pace(results, laps, pits, stints, meta, path, title="", n=10):
    if laps.empty:
        raise ValueError("no lap time data available")

    top = results.head(n)["driverId"].tolist()
    subset = laps[laps["driverId"].isin(top)].dropna(subset=["seconds"])
    if subset.empty:
        raise ValueError("no usable lap times")

    if not pits.empty:
        bad = set()
        for _, stop in pits.iterrows():
            bad.add((stop["driverId"], int(stop["lap"])))
            bad.add((stop["driverId"], int(stop["lap"]) + 1))
        keys = list(zip(subset["driverId"], subset["number"].astype(int)))
        subset = subset[[k not in bad for k in keys]]

    # drop safety car and damaged laps
    cutoff = subset["seconds"].min() * QUICKLAP_THRESHOLD
    subset = subset[subset["seconds"] <= cutoff]

    data, colors, labels = [], [], []
    for drv in top:
        vals = subset.loc[subset["driverId"] == drv, "seconds"]
        if len(vals) < 3:
            continue
        data.append(vals.values)
        colors.append(meta[drv]["color"])
        labels.append(meta[drv]["code"])

    if not data:
        raise ValueError("not enough clean laps to plot")

    fig, ax = plt.subplots(figsize=(11, 6))
    style_axes(ax)
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

    ax.set_title(title)
    ax.set_ylabel("Lap time (s)")
    ax.grid(axis="y", alpha=0.2, color="#666666")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_position_changes(results, laps, pits, stints, meta, path, title=""):
    if laps.empty or "position" not in laps.columns:
        raise ValueError("no per-lap position data available")

    grid = {}
    for _, row in results.iterrows():
        g = row.get("grid")
        if pd.notna(g) and int(g) > 0:
            grid[row["driverId"]] = int(g)
    pit_starts = [meta[d]["code"] for d in meta if d not in grid]
    if pit_starts:
        print(f"grid: no starting slot for {', '.join(pit_starts)} (pit lane start)")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_axes(ax)

    for drv, group in laps.groupby("driverId"):
        if drv not in meta:
            continue
        group = group.sort_values("number")
        x = group["number"].tolist()
        y = group["position"].tolist()
        if drv in grid:
            x = [0] + x
            y = [grid[drv]] + y
        ax.plot(
            x, y,
            label=meta[drv]["code"],
            color=meta[drv]["color"],
            linestyle=meta[drv].get("linestyle", "-"),
            linewidth=1.6,
        )

    n_drivers = max(len(meta), 20)
    ax.set_ylim([n_drivers + 0.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlim(left=0)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Lap (0 = grid)")
    ax.set_ylabel("Position")
    ax.legend(
        bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="small",
        facecolor="#2a2a2a", labelcolor="#dddddd",
    )
    ax.grid(alpha=0.15, color="#666666")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

def fmt_laptime(td):
    if td is None or pd.isna(td):
        return "n/a"
    total = pd.to_timedelta(td).total_seconds()
    return f"{int(total // 60)}:{total % 60:06.3f}"


def build_caption(results, laps, pits, stints, meta, year, race_name):
    lines = [f"## {race_name} {year} - race report", ""]

    try:
        for medal, (_, row) in zip(["🥇", "🥈", "🥉"], results.head(3).iterrows()):
            lines.append(
                f"{medal} **{meta[row['driverId']]['code']}** ({row.get('constructorName', '')})"
            )
        lines.append("")
    except Exception as exc:
        print(f"podium failed: {exc}")

    try:
        fl = results[results.get("fastestLapRank") == 1]
        if len(fl):
            row = fl.iloc[0]
            lines.append(
                f"**Fastest lap** {meta[row['driverId']]['code']} "
                f"{fmt_laptime(row.get('fastestLapTime'))} "
                f"on lap {int(row.get('fastestLapNumber'))}"
            )
    except Exception as exc:
        print(f"fastest lap failed: {exc}")

    try:
        r = results.copy()
        r = r[(r["grid"] > 0) & r["position"].notna()]
        r["gained"] = r["grid"] - r["position"]
        best = r.sort_values("gained", ascending=False).iloc[0]
        if best["gained"] > 0:
            lines.append(
                f"**Biggest gain** {meta[best['driverId']]['code']} "
                f"P{int(best['grid'])} to P{int(best['position'])} "
                f"(+{int(best['gained'])})"
            )
    except Exception as exc:
        print(f"gainer failed: {exc}")

    try:
        winner = results.iloc[0]
        last = int(winner["laps"])
        won = stints[stints["driverId"] == winner["driverId"]] if stints is not None else None

        if won is not None and not won.empty:
            parts = []
            for _, st in won.iterrows():
                start = st.get("lap_start")
                if pd.isna(start):
                    continue
                start = int(start)
                end = st.get("lap_end")
                end = int(end) if pd.notna(end) else last
                parts.append(f"{st['compound'].title()} ({end - start + 1})")
            n_stops = max(len(parts) - 1, 0)  # parts excludes skipped records
            label = "stop" if n_stops == 1 else "stops"
            lines.append(
                f"**Winning strategy** {n_stops}-{label}, " + " → ".join(parts)
            )
        else:
            stops = sorted(pits.loc[pits["driverId"] == winner["driverId"], "lap"].astype(int))
            bounds = [0] + stops + [last]
            lengths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
            plan = " → ".join(str(x) for x in lengths if x > 0)
            label = "stop" if len(stops) == 1 else "stops"
            lines.append(f"**Winning strategy** {len(stops)}-{label}, stints of {plan} laps")
    except Exception as exc:
        print(f"strategy failed: {exc}")

    lines.append("")
    source = "Jolpica-F1 and OpenF1" if stints is not None else "Jolpica-F1"

    return "\n".join(lines)[:1990]

def post_to_discord(webhook_url, caption, image_paths):
    files, handles = {}, []
    try:
        for i, path in enumerate(image_paths):
            fh = open(path, "rb")
            handles.append(fh)
            files[f"files[{i}]"] = (Path(path).name, fh, "image/png")

        resp = requests.post(
            webhook_url,
            data={"payload_json": json.dumps(
                {"content": caption, "username": WEBHOOK_USERNAME}
            )},
            files=files,
            timeout=60,
        )
        if resp.status_code >= 300:
            print(f"discord returned {resp.status_code}: {resp.text[:400]}")
            return False
        print("posted to discord")
        return True
    finally:
        for fh in handles:
            fh.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    Cache.enable_cache(str(CACHE_DIR))
    ergast = Ergast(result_type="pandas", auto_cast=True)

    if args.year and args.round:
        year, rnd, race_name = args.year, args.round, None
    else:
        found = find_recent_race(ergast)
        if not found:
            print("no race finished in the lookback window, nothing to do")
            return 0
        year, rnd, race_name = found

    print(f"loading {year} round {rnd} {race_name or ''}")
    results, laps, pits, stints, meta, fetched_name = load_race(ergast, year, rnd)
    race_name = race_name or fetched_name or f"Round {rnd}"
    print(f"{len(results)} results, {len(laps)} lap records, {len(pits)} pit stops")

    charts = [
        ("stint_strategy.png", chart_stints, f"{race_name} {year} - tyre strategy"),
        ("race_pace.png", chart_race_pace, f"{race_name} {year} - race pace (top 10)"),
        ("position_changes.png", chart_position_changes, f"{race_name} {year} - position changes"),
    ]

    made = []
    for filename, fn, title in charts:
        path = OUT_DIR / filename
        try:
            fn(results, laps, pits, stints, meta, path, title=title)
            made.append(path)
            print(f"built {filename}")
        except Exception as exc:
            print(f"skipping {filename}: {type(exc).__name__}: {exc}")

    caption = build_caption(results, laps, pits, stints, meta, year, race_name)
    print("\n--- caption ---")
    print(caption)
    print("--- end ---\n")

    if args.dry_run:
        print(f"dry run, {len(made)} charts written to {OUT_DIR}/")
        return 0

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("DISCORD_WEBHOOK_URL is not set")
        return 1
    if not made:
        print("no charts were built, not posting")
        return 1

    return 0 if post_to_discord(webhook, caption, made) else 1


if __name__ == "__main__":
    sys.exit(main())
