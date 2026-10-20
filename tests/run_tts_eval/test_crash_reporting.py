"""A failure inside main() must reach the crash handler despite Hydra's sys.exit."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_failure_inside_main_reaches_crash_notification(tmp_path):
    (tmp_path / "broken.yaml").write_text("run_name: crash_probe\n")
    record = tmp_path / "crashed.json"
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
        import json, runpy, sys
        from thinkbooster.utils import telegram

        def notify_crashed(self, **kwargs):
            with open({str(record)!r}, "w") as stream:
                json.dump({{k: str(v) for k, v in kwargs.items()}}, stream)

        telegram.TelegramNotifier.notify_crashed = notify_crashed
        sys.path.insert(0, {str(ROOT / "scripts")!r})
        sys.argv = ["run_tts_eval.py", "--config-path", {str(tmp_path)!r},
                    "--config-name", "broken", "hydra.run.dir={tmp_path / 'run'}"]
        runpy.run_path({str(ROOT / "scripts" / "run_tts_eval.py")!r},
                       run_name="__main__")
        """
        )
    )
    result = subprocess.run(
        [sys.executable, str(driver)],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 1, result.stderr[-2000:]
    crashed = json.loads(record.read_text())
    # The original error, not "1" from Hydra's SystemExit.
    assert "strategy" in crashed["error"]
