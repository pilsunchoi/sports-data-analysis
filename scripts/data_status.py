"""받은 데이터의 수를 센다. 문서에 적는 수치는 이 스크립트의 출력에서 온다.

    python scripts/data_status.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sportsdata import statsbomb as sb  # noqa: E402


def main() -> None:
    m = sb.matches()
    ev = sb.all_events()
    with360 = [i for i in m["match_id"] if (sb.RAW / f"three-sixty/{i}.json").exists()]
    n_frames = sum(sb.frames(i)["event_uuid"].nunique() for i in with360)
    shots = ev[ev["type_name"] == "Shot"]
    in_play = shots[shots["period"] <= 4]
    size = sum(p.stat().st_size for p in sb.RAW.rglob("*.json")) / 1e6

    print(f"경기            {len(m):>9,}  ({m['match_date'].min()} ~ {m['match_date'].max()})")
    print(f"팀              {ev['team_name'].nunique():>9,}")
    print(f"선수            {ev['player_id'].nunique():>9,}")
    print(f"사건            {len(ev):>9,}")
    print(f"  패스          {(ev['type_name'] == 'Pass').sum():>9,}")
    print(f"  슛(승부차기 제외) {len(in_play):>6,}  그 가운데 골 {(in_play['shot_outcome_name'] == 'Goal').sum():,}")
    print(f"  자책골        {(ev['type_name'] == 'Own Goal For').sum():>9,}")
    print(f"360이 있는 경기 {len(with360):>9,}")
    print(f"프리즈 프레임   {n_frames:>9,}")
    print(f"JSON 크기       {size:>9,.0f} MB")


if __name__ == "__main__":
    main()
