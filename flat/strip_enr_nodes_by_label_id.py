"""
Remove all vertices with a given enr-node label id from Gaston .txt, renumber locals.

Default: read ``flat/{train,test}/enr-node/tech.txt`` and write ``tech-mod.txt``
(same folder), removing only vertices whose label id is ``--label-id`` (default:
26389 = ``(date-entity, _)`` in ``vocab_nodes_enr.json``). No edge collapsing.

  python flat/strip_enr_nodes_by_label_id.py

Overwrite ``tech.txt`` in place instead:
  python flat/strip_enr_nodes_by_label_id.py --in-place

Custom sidecar suffix (default ``-mod`` → ``tech-mod.txt``):
  python flat/strip_enr_nodes_by_label_id.py --suffix -custom
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)


def process_transactions(
    lines: list[str],
    drop_label_id: int,
) -> list[str]:
    header_re = re.compile(r"^t\s*#\s*(\d+)\s*$")
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if header_re.match(s):
            if cur is not None:
                blocks.append(cur)
            cur = [s]
            continue
        if cur is None:
            raise ValueError("line before first t #")
        cur.append(s)
    if cur is not None:
        blocks.append(cur)

    out_blocks: list[list[str]] = []
    for block in blocks:
        if not block:
            continue
        header = block[0].strip()
        verts: dict[int, int] = {}
        edges_raw: list[tuple[int, int, int]] = []
        for ln in block[1:]:
            p = ln.split()
            if not p:
                continue
            if p[0] == "v" and len(p) >= 3:
                vid, lid = int(p[1]), int(p[2])
                verts[vid] = lid
            elif p[0] == "e" and len(p) >= 4:
                edges_raw.append((int(p[1]), int(p[2]), int(p[3])))

        remove = {vid for vid, lid in verts.items() if lid == drop_label_id}
        keep_ids = sorted(vid for vid in verts if vid not in remove)
        if not keep_ids:
            continue
        remap = {old: i for i, old in enumerate(keep_ids)}
        new_edges: list[tuple[int, int, int]] = []
        for u, v, eid in edges_raw:
            if u in remove or v in remove:
                continue
            new_edges.append((remap[u], remap[v], eid))
        new_edges.sort(key=lambda t: (t[0], t[1], t[2]))

        out: list[str] = [header]
        for old in keep_ids:
            out.append(f"v {remap[old]} {verts[old]}")
        for u, v, eid in new_edges:
            out.append(f"e {u} {v} {eid}")
        out_blocks.append(out)

    out_lines: list[str] = []
    for i, blk in enumerate(out_blocks):
        hdr = re.sub(r"^t\s*#\s*\d+\s*$", f"t # {i}", blk[0].strip())
        out_lines.append(hdr)
        out_lines.extend(blk[1:])

    return out_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-id", type=int, default=26389, help="vocab_nodes_enr.json node label id to drop")
    parser.add_argument(
        "--suffix",
        type=str,
        default="-mod",
        help="If not --in-place: output is {dir}/tech{suffix}.txt for input .../tech.txt",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite flat/{train,test}/enr-node/tech.txt (same paths as inputs)",
    )
    args = parser.parse_args()

    pairs = [
        os.path.join(PROJECT_ROOT, "flat", "train", "enr-node", "tech.txt"),
        os.path.join(PROJECT_ROOT, "flat", "test", "enr-node", "tech.txt"),
    ]
    for inp in pairs:
        if not os.path.isfile(inp):
            print(f"Skip missing: {inp}")
            continue
        if args.in_place:
            outp = inp
        else:
            base, ext = os.path.splitext(inp)
            outp = f"{base}{args.suffix}{ext}"
        with open(inp, encoding="utf-8") as f:
            raw = f.read().splitlines()
        out_lines = process_transactions(raw, args.label_id)
        parent = os.path.dirname(outp)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(outp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out_lines))
            if out_lines:
                f.write("\n")
        print(f"Wrote {outp} ({len([l for l in out_lines if l.startswith('t ')])} graphs)")


if __name__ == "__main__":
    main()
