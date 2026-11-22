"""독일 분데스리가 7경기의 추적·사건 데이터(Bassek et al., 2025)를 figshare에서 받아 data/raw/idsse/에 둔다.

    python scripts/fetch_idsse.py

파일 이름을 그대로 둔다. 이미 받은 파일은 크기가 맞으면 다시 받지 않는다. 받은 뒤 figshare가 적어 둔
MD5와 대조한다. 위치 파일 7개가 각각 350~420MB이고 전체는 2.6GB다. 라이선스는 CC BY 4.0이다.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/28196177"   # doi:10.6084/m9.figshare.28196177
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "idsse"
CHUNK = 1 << 20


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(item: dict) -> str:
    """받았으면 'new', 있었으면 'kept'. 해시가 다르면 멈춘다."""
    path = DEST / item["name"]
    status = "kept"
    if not (path.exists() and path.stat().st_size == item["size"]):
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(item["download_url"], timeout=300) as r, tmp.open("wb") as f:
            while block := r.read(CHUNK):
                f.write(block)
        tmp.replace(path)
        status = "new"
    want = item.get("supplied_md5") or item.get("computed_md5")
    if want and md5(path) != want:
        raise SystemExit(f"{path.name}: MD5가 figshare의 값과 다르다. 파일을 지우고 다시 받는다.")
    return status


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE, timeout=120) as r:
        article = json.loads(r.read())
    (DEST / "figshare-article.json").write_text(json.dumps(article, ensure_ascii=False, indent=1), encoding="utf-8")
    results = []
    for item in sorted(article["files"], key=lambda f: f["size"]):
        results.append(fetch(item))
        print(f"{results[-1]:5} {item['size'] / 1e6:7.1f}MB  {item['name']}", flush=True)
    print(f"파일 {len(results)}개: " + ", ".join(f"{k} {results.count(k)}" for k in ("new", "kept")))
    print(f"라이선스: {article['license']['name']}. 둔 곳: {DEST}")


if __name__ == "__main__":
    main()
