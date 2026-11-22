"""장의 본문과 표에 쓰는 수의 표기."""

import pandas as pd


def f0(x) -> str:
    return f"{x:,.0f}"


def f1(x, k: int = 1) -> str:
    return f"{x:,.{k}f}"


def pc(x, k: int = 1) -> str:
    """비율을 백분율의 숫자로."""
    return f"{100 * x:.{k}f}"


_FINAL = {"0": "ㅇ", "1": "ㄹ", "3": "ㅁ", "6": "ㄱ", "7": "ㄹ", "8": "ㄹ"}   # 숫자를 읽을 때의 받침. 2, 4, 5, 9는 받침이 없다
_JOSA = {"로": ("로", "으로"), "다": ("다", "이다"), "와": ("와", "과"), "는": ("는", "은"),
         "가": ("가", "이"), "를": ("를", "을"), "고": ("고", "이고")}


def josa(text: str, kind: str) -> str:
    """숫자로 끝나는 글에 조사를 붙인다. kind는 받침이 없을 때의 형태(로, 다, 와, 는, 가, 를, 고)."""
    final = _FINAL.get(text.rstrip("%)")[-1], "")
    plain, closed = _JOSA[kind]
    if kind == "로":
        return text + (plain if final in ("", "ㄹ") else closed)
    return text + (closed if final else plain)


def tab(df: pd.DataFrame, title: str, digits=1) -> None:
    """표를 제목과 함께 마크다운으로 찍는다. digits는 소수 자릿수(수 하나 또는 열마다의 목록)."""
    d = df.copy()
    align = []
    for j, c in enumerate(d.columns):
        k = digits[j] if isinstance(digits, (list, tuple)) else digits
        if pd.api.types.is_bool_dtype(d[c]):
            align.append("left")
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:,}")
            align.append("right")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v, k=k: "" if pd.isna(v) else f"{v:,.{k}f}")
            align.append("right")
        else:
            align.append("left")
    print(f"**{title}**\n")
    print(d.to_markdown(index=False, colalign=align, disable_numparse=True))
    print()
