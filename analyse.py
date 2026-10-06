"""Turn the collector's snapshots into punctuality figures, charts and an event log.

Usage:  python3 analyse.py
Reads:  data/raw-*.jsonl.gz                  (written by collector.py)
Writes: output/departures.csv   one row per departure
        output/event_log.csv    event log for PM4Py
        output/*.png            charts
        findings.md             the summary

Needs:  pip install pandas matplotlib tabulate
All times are Europe/Zurich.
"""
import glob
import gzip
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TZ = "Europe/Zurich"

PUNCTUAL_BELOW = 3       # Swiss federal definition: less than 3 minutes late
MIN_N = 30               # figures resting on fewer departures are flagged
LOOK_BACK = 10           # minutes the collector keeps watching after the scheduled time
CONNECTION_WINDOW = 5    # a train this many minutes after the bus/tram counts as a connection
STATION_TRAINS = "Genève"
STATION_STREET = "Genève, gare Cornavin"
COLOURS = {"Geneva": "#2a78d6", "Zurich": "#eb6834", "Bern": "#1baf7a"}
KEY = ["stop", "line", "to", "scheduled"]   # what identifies one departure


# ---------------------------------------------------------------- loading

def read_snapshots():
    """One row per (snapshot, departure on the board)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "data", "**", "raw-*.jsonl.gz"), recursive=True)):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    for entry in record.get("response", {}).get("stationboard", []):
                        stop = entry["stop"]
                        rows.append({
                            "fetched": record["fetched_at"],
                            "city": record["city"],
                            "stop": record["stop_name"],
                            "category": entry["category"],
                            "line": entry["number"],
                            "to": entry["to"],
                            "scheduled": stop["departure"],
                            "delay": stop["delay"],   # whole minutes, None = no prognosis
                        })
        except EOFError:
            pass  # the collector was killed in the middle of the last line
    df = pd.DataFrame(rows)
    for column in ["fetched", "scheduled"]:
        df[column] = pd.to_datetime(df[column], utc=True).dt.tz_convert(TZ)
    df["type"] = df["category"].map({"T": "tram", "B": "bus"}).fillna("train")
    df["line"] = df["type"] + " " + df["line"].astype(str)
    return df.sort_values("fetched")


def one_row_per_departure(snaps):
    """Keep the last delay seen for each departure.

    GroupBy.last() skips missing values, so 'delay' is the last prognosis we
    ever saw. It stays missing only if the departure never had one.
    """
    dep = snaps.groupby(KEY, as_index=False).agg(
        city=("city", "first"), type=("type", "first"),
        first_seen=("fetched", "min"), last_seen=("fetched", "max"),
        delay=("delay", "last"),
    )
    # Only keep departures we watched to the end: seen at or after the scheduled
    # time, and scheduled long enough ago that the delay can no longer change.
    end = snaps["fetched"].max() - pd.Timedelta(minutes=LOOK_BACK)
    dep = dep[(dep["last_seen"] >= dep["scheduled"]) & (dep["scheduled"] <= end)].copy()
    dep["punctual"] = (dep["delay"] < PUNCTUAL_BELOW).where(dep["delay"].notna())
    dep["hour"] = dep["scheduled"].dt.hour
    return dep


# ---------------------------------------------------------------- statistics

def summarise(dep, by):
    """Punctuality and delay figures per group, with the sample sizes."""
    table = dep.groupby(by).agg(
        departures=("delay", "size"),
        with_delay=("delay", "count"),
        punctual_pct=("punctual", lambda s: 100 * s.dropna().mean()),
        mean_delay=("delay", "mean"),
        median_delay=("delay", "median"),
        p90_delay=("delay", lambda s: s.quantile(0.9)),
    )
    table["missing_pct"] = 100 * (1 - table["with_delay"] / table["departures"])
    table["trust"] = table["with_delay"].map(lambda n: "ok" if n >= MIN_N else "TOO FEW")
    return table.round(1).reset_index()


def worst(dep, by, how_many):
    """Least punctual groups, ranked only among those with at least MIN_N departures."""
    table = summarise(dep, by)
    ranked = table[table["with_delay"] >= MIN_N].sort_values("punctual_pct")
    return ranked.head(how_many), len(table) - len(ranked)


def missed_connections(dep):
    """Bus/tram at the station versus trains scheduled up to 5 minutes later.

    The connection is 'missed' when the bus/tram's predicted departure is later
    than the train's. We only have departures, not arrivals, so the bus/tram's
    departure time stands in for its arrival, and walking time is ignored.
    A train without a prognosis is assumed to leave on schedule.
    """
    street = dep[(dep["stop"] == STATION_STREET) & dep["delay"].notna()]
    trains = dep[dep["stop"] == STATION_TRAINS]
    pairs = street.merge(trains, how="cross", suffixes=("", "_train"))
    gap = (pairs["scheduled_train"] - pairs["scheduled"]).dt.total_seconds() / 60
    pairs = pairs[(gap > 0) & (gap <= CONNECTION_WINDOW)].copy()
    pairs["gap"] = gap
    pairs["missed"] = pairs["delay"] > pairs["gap"] + pairs["delay_train"].fillna(0)
    return pairs


# ---------------------------------------------------------------- event log

def event_log(snaps, dep):
    """One case per departure, in the column names PM4Py expects."""
    snaps = snaps.merge(dep[KEY], on=KEY)   # only departures we watched to the end
    case = snaps["stop"] + " | " + snaps["line"] + " | " + snaps["to"] + " | " \
        + snaps["scheduled"].dt.strftime("%Y-%m-%d %H:%M")
    snaps = snaps.assign(case=case)
    previous = snaps.groupby("case")["delay"].shift()
    first = ~snaps.duplicated("case")

    activity = pd.Series(None, index=snaps.index, dtype=object)
    activity[snaps["delay"] > previous] = "Delay increased"
    activity[snaps["delay"] < previous] = "Delay decreased"
    activity[snaps["delay"].notna() & previous.isna() & ~first] = "Prognosis received"
    activity[snaps["delay"].isna() & previous.notna()] = "Prognosis lost"
    activity[first] = "First seen on board"
    changes = snaps.assign(activity=activity, time=snaps["fetched"]).dropna(subset=["activity"])

    final = snaps.drop_duplicates("case").drop(columns="delay").merge(dep[KEY + ["delay"]], on=KEY)
    scheduled = final.drop(columns="delay").assign(activity="Scheduled departure time", time=final["scheduled"])
    known = final[final["delay"].notna()]
    late = known["delay"] >= PUNCTUAL_BELOW
    departed = known.assign(
        activity=late.map({True: "Departed late", False: "Departed punctual"}),
        time=known["scheduled"] + pd.to_timedelta(known["delay"], unit="min"))
    unknown = final[final["delay"].isna()].drop(columns="delay")
    unknown = unknown.assign(activity="Departed, delay unknown", time=unknown["scheduled"])

    log = pd.concat([changes, scheduled, departed, unknown]).sort_values(["case", "time"], kind="stable")
    log = log.rename(columns={"case": "case:concept:name", "activity": "concept:name",
                              "time": "time:timestamp", "delay": "delay_minutes"})
    return log[["case:concept:name", "concept:name", "time:timestamp", "city", "stop",
                "type", "line", "to", "scheduled", "delay_minutes"]]


# ---------------------------------------------------------------- charts

def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)


def chart_by_city_and_type(by_city_type):
    table = by_city_type[by_city_type["type"] != "train"]
    fig, ax = plt.subplots(figsize=(7, 4))
    types = ["tram", "bus"]
    for i, city in enumerate(COLOURS):
        rows = table[table["city"] == city].set_index("type").reindex(types)
        x = [j + (i - 1) * 0.27 for j in range(len(types))]
        bars = ax.bar(x, rows["punctual_pct"], width=0.25, color=COLOURS[city], label=city)
        for bar, n in zip(bars, rows["with_delay"]):
            if pd.notna(n):
                ax.annotate(f"n={int(n)}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(range(len(types)), types)
    ax.set_ylim(0, 108)
    style(ax, "Share of departures less than 3 minutes late", "% punctual")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.savefig(os.path.join(OUT, "punctuality_by_city.png"), dpi=150, bbox_inches="tight")


def chart_by_hour(by_hour):
    fig, ax = plt.subplots(figsize=(7, 4))
    for city, colour in COLOURS.items():
        rows = by_hour[(by_hour["city"] == city) & (by_hour["with_delay"] >= MIN_N)]
        ax.plot(rows["hour"], rows["punctual_pct"], color=colour, linewidth=2, marker="o", label=city)
    ax.set_xlabel(f"scheduled hour of departure (hours with fewer than {MIN_N} departures left out)")
    ax.set_xticks(sorted(by_hour["hour"].unique()))
    ax.legend(frameon=False)
    style(ax, "Tram and bus punctuality by hour of day", "% punctual")
    fig.savefig(os.path.join(OUT, "punctuality_by_hour.png"), dpi=150, bbox_inches="tight")


def chart_worst_lines(lines):
    if lines.empty:
        return  # no line has MIN_N departures yet
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{line}  (n={n})" for line, n in zip(lines["line"], lines["with_delay"])]
    ax.barh(labels[::-1], lines["punctual_pct"][::-1], color=COLOURS["Geneva"], height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% punctual")
    style(ax, "Geneva: least punctual lines", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    fig.savefig(os.path.join(OUT, "worst_lines_geneva.png"), dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------- findings.md

def md(table):
    return table.to_markdown(index=False) if len(table) else "_No group has enough departures yet._"


def write_findings(snaps, dep, by_city_type, by_stop, by_hour, pairs):
    street = dep[dep["type"] != "train"]
    by_city = summarise(street, "city").set_index("city")
    days = ", ".join(sorted(snaps["fetched"].dt.strftime("%A %d %B %Y").unique()))
    minutes = snaps["fetched"].dt.floor("min").nunique()
    lines = [
        "# Findings: is Geneva less punctual?", "",
        "Generated by `analyse.py`. Punctual means less than 3 minutes late at departure. "
        f"A figure marked TOO FEW rests on fewer than {MIN_N} departures and should not be trusted.", "",
        "## The sample", "",
        f"- Days: {days}",
        f"- From {snaps['fetched'].min():%Y-%m-%d %H:%M} to {snaps['fetched'].max():%Y-%m-%d %H:%M}, "
        f"with snapshots in {minutes} separate minutes (about {minutes / 60:.1f} hours of observation)",
        f"- Hours of day with departures: {dep['hour'].min()}:00 to {dep['hour'].max()}:59",
        f"- {len(dep)} departures, {int(dep['delay'].notna().sum())} of them with a delay value", "",
        "## 1. Is Geneva less punctual than the comparison cities?", "",
        "Tram and bus only, all monitored stops per city:", "",
        md(by_city.reset_index()), "",
        "By vehicle type (trains were only collected in Geneva):", "",
        md(by_city_type), "",
        "Like for like, the stop in front of each main station, tram and bus:", "",
        md(by_stop[by_stop["stop"].isin([STATION_STREET, "Zürich, Bahnhofplatz/HB", "Bern, Bahnhof"])]), "",
        "![Punctuality by city](output/punctuality_by_city.png)", "",
        "## 2. Which lines, stops and hours are worst?", "",
    ]
    for city in COLOURS:
        table, skipped = worst(street[street["city"] == city], "line", 10)
        lines += [f"Worst lines in {city} ({skipped} lines left out for having fewer than "
                  f"{MIN_N} departures):", "", md(table), ""]
    table, skipped = worst(dep[dep["city"] == "Geneva"], "stop", 5)
    lines += [f"Worst stops in Geneva ({skipped} left out):", "", md(table), "",
              "Delay by hour of day, tram and bus:", "", md(by_hour), "",
              "![Worst lines](output/worst_lines_geneva.png)",
              "![By hour](output/punctuality_by_hour.png)", ""]

    delayed = pairs[pairs["delay"] > 0]
    lines += [
        "## 3. Missed connections at Genève station", "",
        f"A connection is a tram or bus at {STATION_STREET} followed by a train scheduled "
        f"1 to {CONNECTION_WINDOW} minutes later.", "",
        f"- Connections in the sample: {len(pairs)}",
        f"- Missed (tram/bus predicted to leave after the train): {int(pairs['missed'].sum())}"
        + (f" ({100 * pairs['missed'].mean():.1f}%)" if len(pairs) else ""),
        f"- Of the {len(delayed)} connections where the tram/bus was at least 1 minute late: "
        f"{int(delayed['missed'].sum())} missed"
        + (f" ({100 * delayed['missed'].mean():.1f}%)" if len(delayed) else ""), "",
        "This is a lower bound. The API gives departures, not arrivals, so the tram/bus departure "
        "time stands in for its arrival, and walking time to the platform is counted as zero.", "",
        "## 4. Is the delay data good enough to drive our processes live?", "",
        "Share of departures that never had a delay value:", "",
        md(by_city_type[["city", "type", "departures", "missing_pct"]]), "",
        "What the API does not provide (checked in the raw responses):", "",
        "- No cancellation marker on any departure.",
        "- `prognosis.platform`, `realtimeAvailability` and the capacity fields are always null.",
        "- Delays come in whole minutes, and the API can be about a minute behind the source.",
        "- A departure leaves the normal board at its scheduled time even when it is late; "
        "a live process has to ask for the board from a few minutes in the past, as the collector does.", "",
        "## Limits of this sample", "",
        "- Delay is measured at departure. The federal figure is measured on arrival, so the "
        "numbers are not directly comparable with official statistics.",
        f"- We stop watching a departure {LOOK_BACK} minutes after its scheduled time, so longer "
        "delays are recorded as the value seen at that point.",
        "- The same vehicle is counted once at every monitored stop it passes.",
        "- Geneva has 7 monitored stops, Zürich and Bern one each.",
        "- A departure whose prognosis is missing is left out of the percentages, never counted as on time.",
    ]
    with open(os.path.join(HERE, "findings.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    snaps = read_snapshots()
    dep = one_row_per_departure(snaps)
    street = dep[dep["type"] != "train"]

    by_city_type = summarise(dep, ["city", "type"])
    by_stop = summarise(street, ["city", "stop"])
    by_hour = summarise(street, ["city", "hour"])
    pairs = missed_connections(dep)

    dep.to_csv(os.path.join(OUT, "departures.csv"), index=False)
    event_log(snaps, dep).to_csv(os.path.join(OUT, "event_log.csv"), index=False)
    chart_by_city_and_type(by_city_type)
    chart_by_hour(by_hour)
    chart_worst_lines(worst(street[street["city"] == "Geneva"], "line", 10)[0])
    write_findings(snaps, dep, by_city_type, by_stop, by_hour, pairs)

    print(by_city_type.to_string(index=False))
    print(f"\n{len(dep)} departures from {len(snaps)} board rows. See findings.md and output/.")


if __name__ == "__main__":
    main()
