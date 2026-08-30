from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_grype_ignore_policy_rejects_expired_accepted_risk(tmp_path: Path) -> None:
    config = tmp_path / ".grype.yaml"
    config.write_text(
        """\
ignore:
  - vulnerability: GHSA-example
    reason: accepted-risk
    package:
      name: example
    expires: "2026-04-16"
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_grype_ignore_expiry.py",
            str(config),
            "--today",
            "2026-08-30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "GHSA-example for example expired on 2026-04-16" in result.stderr
