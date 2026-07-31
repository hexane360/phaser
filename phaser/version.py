import functools
import subprocess
import typing as t
from pathlib import Path

import pane

from . import __version__


class GitInfo(pane.PaneBase):
    """Git state of the checkout phaser is running from"""
    commit: str
    short_commit: str
    dirty: bool
    """Whether tracked files differ from `commit` (untracked files don't count)"""
    branch: t.Optional[str] = None
    """Current branch, or `None` when HEAD is detached"""


class VersionInfo(pane.PaneBase):
    version: str
    git: t.Optional[GitInfo] = None

    def __str__(self) -> str:
        git = ''
        if self.git is not None:
            parts = (self.git.branch, self.git.short_commit, 'dirty' if self.git.dirty else None)
            if any(parts):
                git = f" (git {', '.join(p for p in parts if p)})"
        return f'phaser {self.version}{git}'


@functools.lru_cache(1)
def version_info() -> VersionInfo:
    return VersionInfo(__version__, _git_info())


def _git(*args: str) -> t.Optional[str]:
    # run in phaser's own checkout, not the caller's cwd
    try:
        proc = subprocess.run(
            ('git', *args), cwd=Path(__file__).parent.parent,
            check=True, capture_output=True, text=True, timeout=5.,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    return proc.stdout


def _git_info() -> t.Optional[GitInfo]:
    if (commit := _git('rev-parse', 'HEAD')) is None:
        return None
    commit = commit.strip()
    status = _git('status', '--porcelain', '--untracked-files=no')
    # empty on a detached HEAD
    branch = (_git('rev-parse', '--abbrev-ref', '--symbolic-full-name', 'HEAD') or '').strip()
    return GitInfo(
        commit, commit[:7], dirty=bool(status and status.strip()),
        branch=branch if branch and branch != 'HEAD' else None,
    )

__all__ = [
    'VersionInfo',
    '__version__',
    'version_info',
]
