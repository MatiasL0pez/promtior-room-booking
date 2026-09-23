import os
import re
from collections import defaultdict


def eval_repeat() -> int:
    return int(os.environ.get("EVAL_REPEAT", "1"))


def pytest_generate_tests(metafunc):
    if eval_repeat() > 1:
        metafunc.fixturenames.append("repetition")
        metafunc.parametrize("repetition", range(eval_repeat()), ids=lambda index: f"run{index}")


def pytest_terminal_summary(terminalreporter):
    outcomes_by_case = defaultdict(list)
    for outcome in ("passed", "failed", "error"):
        for report in terminalreporter.stats.get(outcome, []):
            if outcome == "error" and report.when != "setup":
                continue
            if outcome != "error" and report.when != "call":
                continue
            if "evals/" not in report.nodeid.replace("\\", "/"):
                continue
            case = re.sub(r"run\d+-?", "", report.nodeid.split("::")[-1]).replace("[]", "")
            outcomes_by_case[case].append(outcome == "passed")
    if not outcomes_by_case:
        return
    terminalreporter.section(f"eval pass rates (EVAL_REPEAT={eval_repeat()})")
    for case, outcomes in sorted(outcomes_by_case.items()):
        terminalreporter.write_line(f"{sum(outcomes):>3}/{len(outcomes)}  {case}")
