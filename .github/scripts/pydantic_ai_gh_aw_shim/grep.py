"""Claude's `Grep` tool — recursively regex-search workspace files."""

import shutil
import subprocess

from .shared import attach_context, clip, resolve


def grep(pattern: str, path: str = '.') -> str:
    """Recursively regex-search workspace files.

    Uses ripgrep when available, otherwise falls back to GNU grep (always
    present in the AWF sandbox). Returns `file:line:text` matches (capped).
    """
    base = resolve(path)
    try:
        rg = shutil.which('rg')
        if rg:
            r = subprocess.run(
                [rg, '-n', '--no-heading', '-e', pattern, str(base)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        else:
            # rg not available — fall back to GNU grep (always installed)
            grep_bin = shutil.which('grep') or 'grep'
            r = subprocess.run(
                [grep_bin, '-rn', '-P', '--include=*', '-e', pattern, str(base)],
                capture_output=True,
                text=True,
                timeout=60,
            )
    except subprocess.TimeoutExpired:
        return 'error: grep timed out'
    return clip(attach_context(path) + (r.stdout or '(no matches)'))
