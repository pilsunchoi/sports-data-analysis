"""StatsBomb 공개 데이터에서 한 대회의 파일을 받아 data/raw/statsbomb/에 둔다.

    python scripts/fetch_statsbomb.py            2022 남자 월드컵
    python scripts/fetch_statsbomb.py 72 107     2023 여자 월드컵(대회 ID, 시즌 ID)

저장소의 경로와 파일 이름을 그대로 둔다. 이미 받은 파일은 다시 받지 않는다.
"""

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "statsbomb"


def fetch(rel: str) -> str:
    """rel은 저장소 안의 경로다. 받았으면 'new', 있었으면 'kept', 저장소에 없으면 'absent'."""
    path = DEST / rel
    if path.exists() and path.stat().st_size > 0:
        return "kept"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(BASE + rel, timeout=120) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "absent"
        raise
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return "new"


def main(competition: int = 43, season: int = 106) -> None:
    for rel in ("LICENSE.pdf", "data/competitions.json", f"data/matches/{competition}/{season}.json"):
        fetch(rel)
    matches = json.loads((DEST / f"data/matches/{competition}/{season}.json").read_text(encoding="utf-8"))
    rels = [f"data/{kind}/{m['match_id']}.json" for m in matches for kind in ("events", "lineups", "three-sixty")]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(fetch, rels))
    print(f"경기 {len(matches)}개, 파일 {len(rels)}개: "
          + ", ".join(f"{k} {results.count(k)}" for k in ("new", "kept", "absent")))
    print(f"둔 곳: {DEST}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:3]))
