"""렌더된 GitHub용 MD를 손본다. chapters/_quarto.yml의 post-render가 부른다.

pandoc은 원문의 \\~를 맨 ~로 내보낸다. GitHub은 한 줄에 맨 ~가 둘 이상이면 그 사이를
취소선으로 그리므로, 코드와 수식 밖의 ~를 다시 \\~로 바꾼다.
"""

import os
import re
from pathlib import Path

KEEP = re.compile(r"(`[^`]*`|\$\$[^$]*\$\$|\$[^$]+\$)")


def escape_tilde(line: str) -> str:
    parts = KEEP.split(line)
    parts[::2] = [re.sub(r"(?<!\\)~", r"\\~", p) for p in parts[::2]]
    return "".join(parts)


def fix_file(path: Path) -> None:
    out, fenced = [], False
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and not line.startswith("!["):
            line = escape_tilde(line)
        out.append(line)
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")


for f in os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "").split("\n"):
    if f.endswith(".md"):
        fix_file(Path(f))
