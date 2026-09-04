import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import requests
import fastf1
import fastf1.plotting

CACHE_DIR = Path("cache")
OUT_DIR = Path("out")
LOOKBACK_HOURS = 96        
WEBHOOK_USERNAME = "The Paddock"
GREY = "#888888"

def setup():
    CACHE_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        fastf1.plotting.setup_mpl(
            mpl_timedelta_support=True,
            color_scheme="fastf1",
        )
    except TypeError:
        fastf1.plotting.setup_mpl()


def driver_color(abb, session):
    try:
        return fastf1.plotting.get_driver_color(abb, session=session)
    except Exception:
        return GREY


def compound_color(compound, session):
    try:
        return fastf1.plotting.get_compound_color(compound, session=session)
    except Exception:
        return GREY

def find_recent_race(lookback_hours=LOOKBACK_HOURS):
    """Return (year, round_number) for the most recent race that has finished."""
    now = datetime.now(timezone.utc)
    found = None

    for year in (now.year, now.year - 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            print(f"could not load {year} schedule: {exc}")
            continue

        for _, event in schedule.iterrows():
            try:
                race_date = event.get_session_date("Race", utc=True)
            except Exception:
                continue
            if race_date is None or pd.isna(race_date):
                continue
            if race_date.tzinfo is None:
                race_date = race_date.tz_localize("UTC")

            age = now - race_date
            if timedelta(hours=3) < age < timedelta(hours=lookback_hours):
                found = (year, int(event["RoundNumber"]), event["EventName"])

        if found:
            break

    return found

def chart_tyre_strategy(session, path):
    laps = session.laps
    order = [
        session.get_driver(d)["Abbreviation"]
        for d in session.drivers
    ]

    stints = (
        laps[["Driver", "Stint", "Compound", "LapNumber"]]
        .groupby(["Driver", "Stint", "Compound"])
        .count()
        .reset_index()
        .rename(columns={"LapNumber": "StintLength"})
    )

    fig, ax = plt.subplots(figsize=(9, 10))
    for abb in order:
        previous = 0
        for _, row in stints[stints["Driver"] == abb].iterrows():
            ax.barh(
                y=abb,
                width=row["StintLength"],
                left=previous,
                color=compound_color(row["Compound"], session),
                edgecolor="black",
                linewidth=0.6,
            )
            previous += row["StintLength"]

    ax.set_title(f"{session.event['EventName']} {session.event.year} - tyre strategy")
    ax.set_xlabel("Lap")
    ax.invert_yaxis()
    ax.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_race_pace(session, path, n=10):
    top = session.drivers[:n]
    laps = session.laps.pick_drivers(top).pick_quicklaps().reset_index()
    if laps.empty:
        raise ValueError("no quick laps available")

    laps["Seconds"] = laps["LapTime"].dt.total_seconds()
    order = [session.get_driver(d)["Abbreviation"] for d in top]

    data, colors, labels = [], [], []
    for abb in order:
        vals = laps.loc[laps["Driver"] == abb, "Seconds"].dropna()
        if len(vals) < 3:
            continue
        data.append(vals.values)
        colors.append(driver_color(abb, session))
        labels.append(abb)

    fig, ax = plt.subplots(figsize=(11, 6))
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(1.6)

    ax.set_title(f"{session.event['EventName']} {session.event.year} - race pace (top {n})")
    ax.set_ylabel("Lap time (s)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_position_changes(session, path):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for drv in session.drivers:
        drv_laps = session.laps.pick_drivers(drv)
        if drv_laps.empty:
            continue
        abb = drv_laps["Driver"].iloc[0]
        ax.plot(
            drv_laps["LapNumber"],
            drv_laps["Position"],
            label=abb,
            color=driver_color(abb, session),
            linewidth=1.6,
        )

    ax.set_ylim([20.5, 0.5])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Lap")
    ax.set_ylabel("Position")
    ax.set_title(f"{session.event['EventName']} {session.event.year} - position changes")
    ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize="small")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def fmt_laptime(td):
    if pd.isna(td):
        return "n/a"
    total = td.total_seconds()
    return f"{int(total // 60)}:{total % 60:06.3f}"


def build_caption(session):
    results = session.results
    event = session.event
    lines = [f"## {event['EventName']} {event.year} - race report", ""]

    try:
        podium = results.sort_values("Position").head(3)
        placings = ["🥇", "🥈", "🥉"]
        for medal, (_, row) in zip(placings, podium.iterrows()):
            lines.append(f"{medal} **{row['Abbreviation']}** ({row['TeamName']})")
        lines.append("")
    except Exception as exc:
        print(f"podium failed: {exc}")

    try:
        fastest = session.laps.pick_fastest()
        lines.append(
            f"**Fastest lap** {fastest['Driver']} "
            f"{fmt_laptime(fastest['LapTime'])} on lap {int(fastest['LapNumber'])}"
        )
    except Exception as exc:
        print(f"fastest lap failed: {exc}")

    try:
        r = results.copy()
        r = r[(r["GridPosition"] > 0) & r["Position"].notna()]
        r["Gained"] = r["GridPosition"] - r["Position"]
        best = r.sort_values("Gained", ascending=False).iloc[0]
        if best["Gained"] > 0:
            lines.append(
                f"**Biggest gain** {best['Abbreviation']} "
                f"P{int(best['GridPosition'])} to P{int(best['Position'])} "
                f"(+{int(best['Gained'])})"
            )
    except Exception as exc:
        print(f"gainer failed: {exc}")

    try:
        winner = results.sort_values("Position").iloc[0]["Abbreviation"]
        stints = (
            session.laps.pick_drivers(winner)[["Stint", "Compound", "LapNumber"]]
            .groupby(["Stint", "Compound"])
            .count()
            .reset_index()
        )
        strategy = " → ".join(
            f"{row['Compound'].title()} ({int(row['LapNumber'])})"
            for _, row in stints.iterrows()
        )
        lines.append(f"**Winning strategy** {strategy}")
    except Exception as exc:
        print(f"strategy failed: {exc}")

    lines.append("")
    lines.append("Data via FastF1. Argue about it below.")

    caption = "\n".join(lines)
    return caption[:1990]

def post_to_discord(webhook_url, caption, image_paths):
    files, handles = {}, []
    try:
        for i, path in enumerate(image_paths):
            fh = open(path, "rb")
            handles.append(fh)
            files[f"files[{i}]"] = (Path(path).name, fh, "image/png")

        payload = {"content": caption, "username": WEBHOOK_USERNAME}
        resp = requests.post(
            webhook_url,
            data={"payload_json": json.dumps(payload)},
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

    setup()

    if args.year and args.round:
        year, rnd, name = args.year, args.round, None
    else:
        found = find_recent_race()
        if not found:
            print("no race finished in the lookback window, nothing to do")
            return 0
        year, rnd, name = found

    print(f"loading {year} round {rnd} {name or ''}")
    session = fastf1.get_session(year, rnd, "R")
    # telemetry is heavy and none of these charts need it
    session.load(telemetry=False, weather=False, messages=False)

    charts = [
        ("tyre_strategy.png", chart_tyre_strategy),
        ("race_pace.png", chart_race_pace),
        ("position_changes.png", chart_position_changes),
    ]

    made = []
    for filename, fn in charts:
        path = OUT_DIR / filename
        try:
            fn(session, path)
            made.append(path)
            print(f"built {filename}")
        except Exception as exc:
            print(f"skipping {filename}: {exc}")

    caption = build_caption(session)
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
