import pytest

from instruction_drift import evaluation as ev
from instruction_drift.schemas import InstructionCategory, InstructionRecord, Scope, Verification

PY = "```python\n{}\n```"


@pytest.mark.parametrize(
    "text,params,expected",
    [
        ('{"a": 1}', {}, True),
        ('  {"a": 1}\n', {"required_keys": ["a"]}, True),
        ('{"a": 1}', {"required_keys": ["b"]}, False),
        ("[1, 2]", {}, False),
        ("[1, 2]", {"require_object": False}, True),
        ('Here: {"a": 1}', {}, False),
        ('```json\n{"a": 1}\n```', {}, False),
        ('```json\n{"a": 1}\n```', {"allow_fence": True}, True),
        ('{"a": 1,}', {}, False),
    ],
)
def test_json_valid(text, params, expected):
    assert ev.json_valid(text, params).compliance is expected


@pytest.mark.parametrize(
    "text,params,expected",
    [
        (PY.format("x = 1"), {}, True),
        ("```\nprint('hi')\n```", {}, True),
        ("```js\nconsole.log(1)\n```", {}, False),
        ("```\nconst x = 1;\n```", {}, False),
        # Known proxy limit: untagged JS that is also valid Python syntax passes.
        ("```\nconsole.log(1);\n```", {}, True),
        (PY.format("x = 1") + "\n" + "```java\nint x;\n```", {}, False),
        ("no code here", {}, False),
        ("no code here", {"require_code": False}, None),
        ("```python\ndef f(:\n```", {}, True),  # tagged Python with a bug is still Python
    ],
)
def test_python_only(text, params, expected):
    assert ev.python_only(text, params).compliance is expected


@pytest.mark.parametrize(
    "code,expected",
    [
        ("import os\nfrom collections import Counter", True),
        ("import os.path\nfrom __future__ import annotations", True),
        ("from . import sibling", True),
        ("import numpy as np", False),
        ("from requests.adapters import HTTPAdapter", False),
        ("import os, pandas", False),
        ("def f(:\n    import numpy", False),  # unparseable: regex fallback still catches it
        ("import mylib", False),
    ],
)
def test_stdlib_only_imports(code, expected):
    assert ev.stdlib_only_imports(PY.format(code), {}).compliance is expected


def test_stdlib_only_imports_edge_cases():
    assert ev.stdlib_only_imports("prose only", {}).compliance is None
    allowed = {"extra_allowed": ["mylib"]}
    assert ev.stdlib_only_imports(PY.format("import mylib"), allowed).compliance is True
    # A shell block is not Python code, so `pip install numpy` is not an import.
    assert ev.stdlib_only_imports("```bash\npip install numpy\n```", {}).compliance is None
    # Truncated output with an unclosed fence is still inspected.
    assert ev.stdlib_only_imports("```python\nimport numpy\n", {}).compliance is False


def test_max_words_boundary():
    assert ev.max_words("one two three", {"max_words": 3}).compliance is True
    assert ev.max_words("one two three four", {"max_words": 3}).compliance is False
    assert ev.max_words("", {"max_words": 0}).compliance is True


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Assumptions: none.", True),
        ("**Assumptions:** none.", True),
        ("  assumptions: lower case ok", True),
        ("## Assumptions: x", True),
        ("My assumptions: x", False),
        ("Answer first. Assumptions: x", False),
    ],
)
def test_starts_with_marker(text, expected):
    assert ev.starts_with_marker(text, {"marker": "Assumptions:"}).compliance is expected


def test_executable_python():
    assert ev.executable_python(PY.format("print(1)"), {}).compliance is True
    assert ev.executable_python(PY.format("def f(:"), {}).compliance is False
    assert ev.executable_python("just prose", {}).compliance is False
    assert ev.executable_python("```js\nx\n```", {}).compliance is False


def test_preserve_function_names():
    p = {"function_names": ["load_data", "parse_row"]}
    both = PY.format("def load_data():\n    pass\n\nasync def parse_row(r):\n    pass")
    assert ev.preserve_function_names(both, p).compliance is True
    renamed = PY.format("def load_data():\n    pass\n\ndef parse_line(r):\n    pass")
    res = ev.preserve_function_names(renamed, p)
    assert res.compliance is False and "parse_row" in res.evidence
    broken = PY.format("def load_data(:\n    pass\ndef parse_row(r):\n    pass")
    assert ev.preserve_function_names(broken, p).compliance is True


def _instr(name, params, version="1.0"):
    return InstructionRecord(
        instruction_id="I1", group_id="G", conversation_id="C", original_text="x",
        normalized_text="x", compressed_text="x", category=InstructionCategory.LENGTH,
        priority=1.0, scope=Scope(kind="persistent", start_turn=0),
        verification=Verification(method="deterministic", evaluator_name=name,
                                  evaluator_version=version, params=params),
        created_at_turn=0, original_token_start=0,
    )


def _evaluate(instr, text, applicable=True):
    return ev.evaluate(instr, text, applicable=applicable, run_id="R", conversation_id="C",
                       checkpoint_id="CP", response_id="RESP", policy_name="B0")


def test_evaluate_builds_versioned_compliance_record():
    rec = _evaluate(_instr("max_words", {"max_words": 2}), "a b c")
    assert (rec.compliance, rec.applicable, rec.failure_type) == (0, True, "too_long")
    assert rec.evaluator_name == "max_words" and rec.evaluator_version == "1.0"

    skipped = _evaluate(_instr("max_words", {"max_words": 2}), "a b c", applicable=False)
    assert skipped.applicable is False and skipped.compliance is None

    vacuous = _evaluate(_instr("stdlib_only_imports", {}), "no code")
    assert vacuous.applicable is False and vacuous.compliance is None

    with pytest.raises(ValueError):
        _evaluate(_instr("max_words", {"max_words": 2}, version="0.9"), "a")
