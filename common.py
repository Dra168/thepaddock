"""
The Paddock - shared plumbing for session reports.

Everything here talks to OpenF1 (https://openf1.org). Unlike the race report,
which needs Jolpica for classified results, practice / qualifying / sprint data
is only available from OpenF1, and its /session_result endpoint covers every
session type in one shape.

OpenF1 serves GitHub Actions runners; F1's own livetiming API does not, which is
why nothing here touches FastF1.

Historical data starts at 2023.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a CI runner
import matplotlib.pyplot as plt
import pandas as pd
import requests

OPENF1_BASE = "https://api.openf1.org/v1"
TIMEOUT = 25
OUT_DIR = Path("out")
WEBHOOK_USERNAME = "The Paddock"

BG = "#1e1e1e"
FG = "#dddddd"
GREY = "#888888"

COMPOUND_COLORS = {
    "SOFT": "#DA291C",
    "MEDIUM": "#FFD12E",
    "HARD": "#F0F0EC",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
    "UNKNOWN": GREY,
}

# Used only when OpenF1 gives no team_colour for a driver.
FALLBACK_COLORS = [
    "#8E7CC3", "#D9A441", "#5FA8D3", "#C36B6B",
    "#7FB069", "#B5838D", "#9C89B8", "#E0A458",
]


# ---------------------------------------------------------------- api


def openf1(endpoint, **params):
    """GET an OpenF1 endpoint. Returns a list; [] on any failure."""
    try:
        resp = requests.get(f"{OPENF1_BASE}/{endpoint}", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"openf1: {endpoint} failed ({exc})")
        return []


def find_recent_session(session_name, lookback_hours=96):
    """Most recent session of this name that finished inside the window.

    Returns the session dict, or None. Searches this year then last year, so it
    still works in the first days of January.
    """
    now = datetime.now(timezone.utc)
    for year in (now.year, now.year - 1):
        sessions = openf1("sessions", year=year, session_name=session_name)
        best = None
        for sess in sessions:
            if sess.get("is_cancelled"):
                continue
            end = sess.get("date_end")
            if not end:
                continue
            try:
                ended = pd.Timestamp(end)
            except Exception:
                continue
            if ended.tzinfo is None:
                ended = ended.tz_localize("UTC")
            age = now - ended
            if timedelta(minutes=20) < age < timedelta(hours=lookback_hours):
                best = sess
        if best:
            return best
    return None


def find_recent_sessions(session_names, lookback_hours=22):
    """Every session in `session_names` that finished inside the window.

    Used by the practice report, which posts one message per practice DAY
    rather than per session: a Friday run picks up FP1 and FP2 together, a
    Saturday run picks up FP3 alone, and on a sprint weekend the Friday run
    picks up FP1 by itself because FP2 and FP3 do not exist.

    Results are restricted to a single meeting, so a stale session from the
    previous round can never be mixed in, and are returned in session order.
    """
    now = datetime.now(timezone.utc)
    found = []
    for year in (now.year, now.year - 1):
        for name in session_names:
            for sess in openf1("sessions", year=year, session_name=name):
                if sess.get("is_cancelled"):
                    continue
                end = sess.get("date_end")
                if not end:
                    continue
                try:
                    ended = pd.Timestamp(end)
                except Exception:
                    continue
                if ended.tzinfo is None:
                    ended = ended.tz_localize("UTC")
                age = now - ended
                if timedelta(minutes=20) < age < timedelta(hours=lookback_hours):
                    found.append((ended, sess))
        if found:
            break

    if not found:
        return []

    # keep only the most recent meeting, in case a window ever spans two
    latest_meeting = max(found, key=lambda x: x[0])[1].get("meeting_key")
    found = [f for f in found if f[1].get("meeting_key") == latest_meeting]
    found.sort(key=lambda x: x[0])
    return [sess for _, sess in found]


def find_sessions(year, session_names, round_hint=None):
    """All sessions of these names for one meeting, for manual runs."""
    out = []
    for name in session_names:
        sess = find_session(year, name, round_hint)
        if sess:
            out.append(sess)
    if not out:
        return []
    latest = out[-1].get("meeting_key")
    out = [s for s in out if s.get("meeting_key") == latest]
    out.sort(key=lambda s: str(s.get("date_start")))
    return out


def find_session(year, session_name, round_hint=None):
    """Find a session by year and name, optionally narrowed by a country or
    circuit name fragment. Returns the session dict or None."""
    sessions = openf1("sessions", year=year, session_name=session_name)
    if not sessions:
        return None
    if round_hint:
        hint = str(round_hint).lower()
        matches = [
            s for s in sessions
            if hint in str(s.get("country_name", "")).lower()
            or hint in str(s.get("circuit_short_name", "")).lower()
            or hint in str(s.get("location", "")).lower()
        ]
        if matches:
            return matches[-1]
        print(f"openf1: no {session_name} matching '{round_hint}' in {year}")
        return None
    return sessions[-1]


def load_drivers(session_key):
    """Map driver_number -> {code, team, color, linestyle}.

    OpenF1 supplies team_colour per driver, so unlike the race report there is
    no hardcoded palette to keep up to date. Teammates share a colour, so the
    second driver of each team gets a dashed line.
    """
    rows = openf1("drivers", session_key=session_key)
    meta, seen_team = {}, {}
    for i, row in enumerate(rows):
        num = row.get("driver_number")
        if num is None:
            continue
        team = row.get("team_name") or ""
        colour = row.get("team_colour")
        color = f"#{colour}" if colour else FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        nth = seen_team.get(team, 0)
        seen_team[team] = nth + 1
        meta[int(num)] = {
            "code": row.get("name_acronym") or str(num),
            "name": row.get("full_name") or "",
            "team": team,
            "color": color,
            "linestyle": ["-", "--", ":"][nth % 3],
        }
    return meta


def session_results(session_key):
    """DataFrame of /session_result, sorted by position.

    `duration` and `gap_to_leader` are scalars for practice, sprint and race,
    but three-element lists for qualifying (Q1/Q2/Q3).
    """
    rows = openf1("session_result", session_key=session_key)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "position" in df.columns:
        df = df.sort_values("position", na_position="last")
    return df.reset_index(drop=True)


def session_laps(session_key):
    """DataFrame of /laps with a `seconds` column, or empty."""
    rows = openf1("laps", session_key=session_key)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["seconds"] = pd.to_numeric(df.get("lap_duration"), errors="coerce")
    return df


def session_stints(session_key):
    """DataFrame of /stints with compound upper-cased, or empty."""
    rows = openf1("stints", session_key=session_key)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["compound"] = (
        df.get("compound", pd.Series(dtype=str))
        .fillna("UNKNOWN").astype(str).str.upper()
    )
    return df


# ---------------------------------------------------------------- values


def seg_value(value, index):
    """Pull one segment out of a qualifying array, or None.

    In qualifying OpenF1 gives duration and gap_to_leader as [Q1, Q2, Q3];
    for other sessions it is a plain number.
    """
    if isinstance(value, (list, tuple)):
        if index < len(value):
            v = value[index]
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else v
        return None
    return value if index == 0 else None


def scalar_value(value):
    """Best available number from a scalar or a qualifying array."""
    if isinstance(value, (list, tuple)):
        vals = [v for v in value if isinstance(v, (int, float)) and not pd.isna(v)]
        return min(vals) if vals else None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return value
    return None


def fmt_time(seconds):
    """78.412 -> 1:18.412"""
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return "n/a"
    seconds = float(seconds)
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def fmt_gap(seconds):
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return ""
    return f"+{float(seconds):.3f}"


# ---------------------------------------------------------------- charts


def style_axes(ax):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color("#ffffff")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def compound_legend(ax, compounds):
    keys = [c for c in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")
            if c in compounds]
    if not keys:
        return
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMPOUND_COLORS[c]) for c in keys]
    ax.legend(handles, [c.title() for c in keys], loc="lower right",
              facecolor="#2a2a2a", labelcolor=FG, fontsize="small")


# ---------------------------------------------------------------- output


def session_title(sess):
    name = sess.get("session_name", "Session")
    where = sess.get("country_name") or sess.get("circuit_short_name") or ""
    year = sess.get("year", "")
    return f"{where} {year} - {name}".strip()


def post_to_discord(caption, image_paths, dry_run=False):
    OUT_DIR.mkdir(exist_ok=True)
    print("\n--- caption ---")
    print(caption)
    print("--- end ---\n")

    if dry_run:
        print(f"dry run, {len(image_paths)} chart(s) written to {OUT_DIR}/")
        return 0

    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("DISCORD_WEBHOOK_URL is not set")
        return 1
    if not image_paths:
        print("no charts were built, not posting")
        return 1

    files, handles = {}, []
    try:
        for i, path in enumerate(image_paths):
            fh = open(path, "rb")
            handles.append(fh)
            files[f"files[{i}]"] = (Path(path).name, fh, "image/png")
        resp = requests.post(
            webhook,
            data={"payload_json": json.dumps(
                {"content": caption[:1990], "username": WEBHOOK_USERNAME})},
            files=files,
            timeout=60,
        )
        if resp.status_code >= 300:
            print(f"discord returned {resp.status_code}: {resp.text[:400]}")
            return 1
        print("posted to discord")
        return 0
    finally:
        for fh in handles:
            fh.close()


def build_charts(specs):
    """Run each (filename, fn) and collect the ones that worked.

    One failing chart must not take the whole post down with it.
    """
    OUT_DIR.mkdir(exist_ok=True)
    made = []
    for filename, fn in specs:
        path = OUT_DIR / filename
        try:
            fn(path)
            made.append(path)
            print(f"built {filename}")
        except Exception as exc:
            print(f"skipping {filename}: {type(exc).__name__}: {exc}")
    return made


def resolve_session(session_name, args):
    """Pick the session from CLI args, or auto-detect the most recent one."""
    if args.year:
        sess = find_session(args.year, session_name, args.round)
        if not sess:
            print(f"no {session_name} found for {args.year} {args.round or ''}")
            return None
    else:
        sess = find_recent_session(session_name, args.lookback)
        if not sess:
            print(f"no {session_name} finished in the last {args.lookback}h, "
                  f"nothing to do")
            return None
    print(f"session {sess.get('session_key')}: {session_title(sess)} "
          f"({sess.get('date_start')})")
    return sess


def add_common_args(parser, session_name):
    parser.add_argument("--year", type=int,
                        help="season; omit to auto-detect the most recent session")
    parser.add_argument("--round", type=str,
                        help="country, circuit or location fragment, e.g. Hungary")
    parser.add_argument("--lookback", type=int, default=96,
                        help="hours to look back when auto-detecting")
    parser.add_argument("--dry-run", action="store_true")
    return parser
