"""Poll Swiss station boards and append every raw response to a JSONL file.

Usage:
    python3 collector.py              # run until Ctrl-C
    python3 collector.py --minutes 10 # stop after 10 minutes

Output: data/raw-YYYY-MM-DD.jsonl.gz, one line per request:
    {"fetched_at": ..., "stop_id": ..., "stop_name": ..., "city": ...,
     "status": 200, "response": {...the API response, unchanged...}}
Failed requests are logged too, with "error" instead of "response".

Only the Python standard library is used.
"""
import argparse
import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

API = "https://transport.opendata.ch/v1/stationboard"
ZURICH = ZoneInfo("Europe/Zurich")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# (stop id, name, city) - ids come from /v1/locations, see phase1/
STOPS = [
    ("8501008", "Genève", "Geneva"),                      # trains
    ("8587057", "Genève, gare Cornavin", "Geneva"),       # tram + bus at the station
    ("8592801", "Genève, Chantepoulet", "Geneva"),
    ("8592812", "Genève, Coutance", "Geneva"),
    ("8587387", "Genève, Bel-Air", "Geneva"),
    ("8587907", "Genève, Plainpalais", "Geneva"),
    ("8587061", "Genève, Rive", "Geneva"),
    ("8587348", "Zürich, Bahnhofplatz/HB", "Zurich"),
    ("8576646", "Bern, Bahnhof", "Bern"),
]

POLL_SECONDS = 60    # each stop is asked once per round
LIMIT = 60           # departures per board
LOOK_BACK_MINUTES = 10  # see fetch()
MAX_BACKOFF = 600    # never wait longer than 10 minutes after errors


def fetch(stop_id):
    """Return (status, parsed JSON or None, error text or None).

    The board normally drops a departure once its scheduled minute has passed,
    even if the vehicle is late and still at the stop. Asking for the board as
    of 10 minutes ago keeps late departures visible while their delay grows.
    """
    since = datetime.now(ZURICH) - timedelta(minutes=LOOK_BACK_MINUTES)
    query = {"id": stop_id, "limit": LIMIT, "datetime": since.strftime("%Y-%m-%d %H:%M")}
    url = API + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": "DTU-02268-student-project"})
    try:
        with urllib.request.urlopen(request, timeout=20) as reply:
            return reply.status, json.loads(reply.read()), None
    except urllib.error.HTTPError as e:
        return e.code, None, str(e)
    except Exception as e:  # timeout, no network, invalid JSON, ...
        return None, None, repr(e)


def append(record):
    """Append one record as its own gzip block, in a single write.

    Each line is a complete gzip member, so stopping the collector at any
    moment leaves a file that still opens with gzip.open().
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    day = record["fetched_at"][:10]
    path = os.path.join(DATA_DIR, f"raw-{day}.jsonl.gz")
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "ab") as f:
        f.write(gzip.compress(line.encode("utf-8")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, help="stop after this many minutes")
    args = parser.parse_args()
    stop_at = time.time() + args.minutes * 60 if args.minutes else None

    gap = POLL_SECONDS / len(STOPS)  # spread the requests evenly over the round
    backoff = 0
    print(f"Polling {len(STOPS)} stops, one request every {gap:.1f}s. Ctrl-C to stop.")
    try:
        while True:
            for stop_id, name, city in STOPS:
                if stop_at and time.time() >= stop_at:
                    return
                started = time.time()
                status, response, error = fetch(stop_id)
                record = {
                    "fetched_at": datetime.now(ZURICH).isoformat(timespec="seconds"),
                    "stop_id": stop_id, "stop_name": name, "city": city, "status": status,
                }
                if error:
                    record["error"] = error
                    backoff = min(max(2 * backoff, 30), MAX_BACKOFF)
                    print(f"{record['fetched_at']} {name}: {error} - pausing {backoff}s")
                else:
                    record["response"] = response
                    backoff = 0
                append(record)
                time.sleep(max(gap - (time.time() - started), 0) + backoff)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopped.")


if __name__ == "__main__":
    main()
