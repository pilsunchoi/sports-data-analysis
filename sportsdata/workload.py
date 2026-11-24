"""추적 데이터에서 선수의 구간별 활동량 표를 만든다. 운동학 2장이 사용한다.

속도는 제공자의 값이다(운동학 1장 V장). 구간은 전반과 후반의 처음 45분을 같은 길이로
나눈 것이고 추가 시간은 쓰지 않는다.
"""

import numpy as np
import pandas as pd

from . import idsse

V_HSR, V_SPR = 19.8, 25.2          # 고속 주행과 스프린트의 문턱(km/h)
HALF_FRAMES = 45 * 60 * idsse.HZ   # 한 구간에서 쓰는 프레임 수


def _build(minutes: int) -> pd.DataFrame:
    step = minutes * 60 * idsse.HZ
    per_half = HALF_FRAMES // step
    rows = []
    for mid in idsse.match_ids():
        info = idsse.matches().set_index("match_id").loc[mid]
        pl = idsse.players(mid).set_index("person_id")
        gl = idsse.goals(mid)
        of = idsse.outfield(mid)
        for h, (sec, d) in enumerate(idsse.tracking(mid).items()):
            o = of[of["section"] == sec]
            kmh = np.round(d["s"][o["row"].to_numpy()].astype(float), 2)
            for b in range(per_half):
                sl = slice(b * step, (b + 1) * step)
                v, alive = kmh[:, sl], d["alive"][sl]
                on = np.isfinite(v)
                m = np.where(on, v, 0.0) / 3.6 * idsse.DT          # 프레임마다의 이동 거리(m)
                before = gl[(gl["section"] == sec) & (gl["seconds"] < b * minutes * 60)]
                prior = gl[gl["section"] == "firstHalf"] if h else gl.iloc[:0]
                done = pd.concat([prior, before])
                home_lead = int((done["team_id"] == info["home_id"]).sum() - (done["team_id"] == info["guest_id"]).sum())
                for i, p in enumerate(o.itertuples()):
                    if not on[i].any():
                        continue
                    side = pl.at[p.person_id, "side"]
                    rows.append({"match_id": mid, "team_id": pl.at[p.person_id, "team_id"], "person_id": p.person_id, "name": p.name,
                                 "position": p.position, "starter": bool(pl.at[p.person_id, "starting"]), "half": h,
                                 "interval": h * per_half + b, "frames": int(on[i].sum()), "full": bool(on[i].all()),
                                 "dist": m[i].sum(), "hsr": m[i][v[i] >= V_HSR - 1e-6].sum(), "spr": m[i][v[i] >= V_SPR - 1e-6].sum(),
                                 "dist_alive": m[i][alive].sum(), "hsr_alive": m[i][alive & (v[i] >= V_HSR - 1e-6)].sum(), "alive_frames": int((alive & on[i]).sum()),
                                 "alive_share": float(alive.mean()), "lead": home_lead if side == "home" else -home_lead})
    return pd.DataFrame(rows)


def intervals(minutes: int = 15) -> pd.DataFrame:
    """선수, 경기, 구간마다 한 행. 그 구간에 경기장에 있던 선수만 있다. 골키퍼도 포함한다.

    frames는 경기장에 있던 프레임 수, full은 구간 내내 있었는지, dist·hsr·spr은 이동 거리와
    고속 주행 거리와 스프린트 거리(m), dist_alive와 hsr_alive는 경기가 진행 중일 때의 이동 거리와 고속 주행 거리, alive_frames는 그때의 프레임 수,
    alive_share는 그 구간에서 경기가 진행 중이던 시간의 비율, lead는 구간이 시작할 때
    그 선수의 팀이 앞선 골 수다.
    """
    path = idsse.CACHE / f"idsse-intervals-{minutes}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    out = _build(minutes)
    out.to_pickle(path)
    return out
