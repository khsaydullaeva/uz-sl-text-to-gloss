#!/usr/bin/env python3
"""Run inside the pose conda env to bridge ui.py to the sibling repo's pose pipeline.

record/tools/text_to_pose.py (in the uzbek-sign-language checkout) only takes
source text on its CLI — it derives its own gloss internally via a dictionary
lookup, and has no --gloss or --report-json flags. This driver imports its
generate_pose()/render() functions directly instead, so we can pass the exact
output paths ui.py expects and get the structured report dict back as JSON.
"""
import json
import sys
from pathlib import Path


def main() -> None:
    pose_repo_root, text, pose_path, mp4_path, report_path = sys.argv[1:6]
    # Optional 6th arg: path to a {label: sign_id} JSON produced by the
    # word-sense disambiguation step. Absent/empty → default sense selection.
    overrides: dict[str, str] = {}
    if len(sys.argv) > 6 and sys.argv[6]:
        overrides_path = Path(sys.argv[6])
        if overrides_path.is_file():
            overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(pose_repo_root) / "record" / "tools"))
    from text_to_pose import generate_pose
    from visualize_pose import render as render_pose

    out_path, report = generate_pose(
        text, output_path=Path(pose_path), sense_overrides=overrides or None
    )
    render_pose(str(out_path), mp4_path, None, "skeleton", 1)
    Path(report_path).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
