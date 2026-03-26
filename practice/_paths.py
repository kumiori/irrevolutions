from pathlib import Path


PRACTICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRACTICE_DIR.parent


def practice_path(*parts):
    return PRACTICE_DIR.joinpath(*parts)


def repo_path(*parts):
    return REPO_ROOT.joinpath(*parts)
