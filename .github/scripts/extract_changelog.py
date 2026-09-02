#!/usr/bin/env python3
"""Extract one version's section from CHANGELOG.md, for use as a release body."""

import re
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <version> <output path>", file=sys.stderr)
    sys.exit(2)

version, out_path = sys.argv[1], Path(sys.argv[2])
changelog = Path('CHANGELOG.md').read_text()

# '## <version>', optionally followed by a date or other annotation
start = re.search(rf'^## +{re.escape(version)}\b.*$', changelog, re.MULTILINE)
if start is None:
    print(f"No '## {version}' section found in CHANGELOG.md", file=sys.stderr)
    sys.exit(1)

end = re.search(r'^## ', changelog[start.end():], re.MULTILINE)
body = changelog[start.end():(start.end() + end.start()) if end else len(changelog)]

out_path.write_text(body.strip() + '\n')
