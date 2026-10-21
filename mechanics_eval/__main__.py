"""Run the current Level 1 mechanics protocol from the repository root."""
import subprocess
import sys
from .runner import arguments, run

if __name__ == '__main__':
    try:
        run(arguments())
    except (ValueError, subprocess.CalledProcessError) as exc:
        sys.exit('Validation failed: ' + str(exc))
