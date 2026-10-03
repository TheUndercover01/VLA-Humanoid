"""Download the Franka Panda MuJoCo model (Apache-2.0) from mujoco_menagerie.

    python -m sim.fetch_assets
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

DEST = Path(__file__).parent / "assets" / "franka_emika_panda"


def main():
    if (DEST / "panda.xml").exists():
        print(f"already present: {DEST}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/google-deepmind/mujoco_menagerie.git", tmp], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "franka_emika_panda"], check=True)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(tmp) / "franka_emika_panda", DEST)
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
