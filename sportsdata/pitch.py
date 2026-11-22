"""경기장 그림과 그림의 공통 양식.

경기장의 선은 StatsBomb 명세서(Open Data Events v4.0.0, 부록 2)의 좌표를 따른다. 원점은
왼쪽 위이고 y는 아래로 커지므로 y축을 뒤집어 그린다.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle


def use_style() -> None:
    """흑백, 맑은 고딕, 300dpi. 위와 오른쪽 축선을 없앤다."""
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "font.size": 8, "axes.unicode_minus": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "grid.color": "0.87", "grid.linewidth": 0.4,
        "legend.frameon": False,
    })


def draw_pitch(ax, half: bool = False, lw: float = 0.7, color: str = "black"):
    """ax에 경기장을 그린다. half=True면 공격하는 쪽 절반(x가 60 이상)만 그린다."""
    kw = dict(fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(Rectangle((0, 0), 120, 80, **kw))
    ax.plot([60, 60], [0, 80], color=color, linewidth=lw)
    ax.add_patch(Circle((60, 40), 10, **kw))
    ax.add_patch(Circle((60, 40), 0.3, color=color))
    for side in (0, 1):                                  # 0은 왼쪽, 1은 오른쪽 골대 쪽
        gx = 120 * side
        s = -1 if side else 1
        ax.add_patch(Rectangle((gx, 18), s * 18, 44, **kw))        # 페널티 구역
        ax.add_patch(Rectangle((gx, 30), s * 6, 20, **kw))         # 골 구역
        ax.add_patch(Rectangle((gx, 36), -s * 2, 8, **kw))         # 골대
        ax.add_patch(Circle((gx + s * 12, 40), 0.3, color=color))  # 페널티 마크
        mid = 180 * side
        ax.add_patch(Arc((gx + s * 12, 40), 20, 20, theta1=mid - 53.13, theta2=mid + 53.13,
                         edgecolor=color, linewidth=lw))
    ax.set_xlim((58, 123) if half else (-3, 123))
    ax.set_ylim(82, -2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def draw_pitch_m(ax, lw: float = 0.6, color: str = "0.35", length: float = 105.0, width: float = 68.0):
    """중앙이 원점인 미터 좌표의 경기장. 추적 데이터(IDSSE)의 좌표에 맞춘다. 선의 치수는 경기 규칙을 따른다."""
    kw = dict(fill=False, edgecolor=color, linewidth=lw)
    hx, hy = length / 2, width / 2
    ax.add_patch(Rectangle((-hx, -hy), length, width, **kw))
    ax.plot([0, 0], [-hy, hy], color=color, linewidth=lw)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * hx, -20.16), -s * 16.5, 40.32, **kw))      # 페널티 구역
        ax.add_patch(Rectangle((s * hx, -9.16), -s * 5.5, 18.32, **kw))        # 골 구역
        mid = 180 if s > 0 else 0
        ax.add_patch(Arc((s * (hx - 11), 0), 18.3, 18.3, theta1=mid - 53.13, theta2=mid + 53.13, edgecolor=color, linewidth=lw))
    ax.set_xlim(-hx - 2, hx + 2)
    ax.set_ylim(-hy - 2, hy + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
