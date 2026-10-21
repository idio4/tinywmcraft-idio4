"""Optional local artifacts are never bundled with the tests."""
import atexit
import os
from pathlib import Path
from tempfile import TemporaryDirectory

PACKAGE = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get('MECHANICS_EVAL_TEST_RUN', PACKAGE/'runs/test-input'))
_temporary = TemporaryDirectory(prefix='mechanics-eval-tests-')
atexit.register(_temporary.cleanup)
OUTPUT = Path(_temporary.name)
