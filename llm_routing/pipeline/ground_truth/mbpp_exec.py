"""Extract Python solutions and evaluate them against MBPP tests.

Warning: this module executes model-generated Python. Run it only inside an
OS-level sandbox or an isolated container with no sensitive mounted data.
Python's ``-I`` flag isolates imports; it is not a security boundary.
"""

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path


FENCE = chr(96) * 3


def _strip_main_test_harness(code):
    """Remove top-level model-added __main__ self-tests."""
    tree = ast.parse(code)
    kept = []

    for node in tree.body:
        is_main_guard = (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        )

        if not is_main_guard:
            kept.append(node)

    tree.body = kept
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def extract_python_code(response):
    """Extract the most plausible syntactically valid Python solution."""
    response = str(response).strip()

    # 1. Prefer a fenced Markdown Python block.
    escaped_fence = re.escape(FENCE)

    pattern = (
        escaped_fence
        + r"(?:python|py)?\s*(.*?)"
        + escaped_fence
    )

    blocks = re.findall(
        pattern,
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if blocks:
        code = max(blocks, key=len).strip()

        try:
            ast.parse(code)
            return _strip_main_test_harness(code)
        except SyntaxError:
            return None

    # 2. Remove markers emitted by some model formats.
    response = (
        response
        .replace("[BEGIN]", "")
        .replace("[DONE]", "")
        .strip()
    )

    # 3. The complete response may be raw Python.
    try:
        ast.parse(response)
        return _strip_main_test_harness(response)
    except SyntaxError:
        pass

    # 4. Try raw Python that follows an initial prose explanation.
    lines = response.splitlines()

    starters = (
        "def ",
        "class ",
        "import ",
        "from ",
        "@",
    )

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if not stripped.startswith(starters):
            continue

        candidate = "\n".join(lines[i:]).strip()

        try:
            ast.parse(candidate)
            return _strip_main_test_harness(candidate)
        except SyntaxError:
            continue

    return None


def run_mbpp_tests(
    code,
    test_list,
    test_setup_code="",
    timeout=5,
):
    """Execute a candidate and MBPP assertions in a temporary directory."""
    if not code:
        return False, "no_code"

    parts = []

    if test_setup_code:
        parts.append(test_setup_code)

    parts.append(code)
    parts.extend(test_list)

    program = "\n\n".join(parts) + "\n"

    with tempfile.TemporaryDirectory(
        prefix="mbpp_eval_"
    ) as tmpdir:

        candidate_path = Path(tmpdir) / "candidate.py"

        candidate_path.write_text(
            program,
            encoding="utf-8",
        )

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(candidate_path),
                ],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        except subprocess.TimeoutExpired:
            return False, "timeout"

        if result.returncode == 0:
            return True, "passed"

        stderr = result.stderr or ""

        if "AssertionError" in stderr:
            return False, "failed_test"

        return False, "runtime_error"


def evaluate_mbpp(
    response,
    test_list,
    test_setup_code="",
    timeout=5,
):
    """Return whether extracted candidate code passes every MBPP test."""
    code = extract_python_code(response)

    passed, _ = run_mbpp_tests(
        code=code,
        test_list=test_list,
        test_setup_code=test_setup_code,
        timeout=timeout,
    )

    return passed
