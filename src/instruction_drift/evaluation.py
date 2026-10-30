"""Deterministic compliance evaluators.

Each evaluator judges one atomic instruction against one response and returns
compliant / violated / not-evaluable (None) plus evidence. Evaluators see only the
response text and the instruction's declared parameters: never risk scores, policy
names, or controller actions.

Known proxies (documented limits, not bugs):
- "Executable code" is checked by `ast.parse`, not by running the code.
- "Python only" accepts a block tagged python/py, or an untagged block that parses.
- The stdlib list comes from `sys.stdlib_module_names` of the running interpreter.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass

from instruction_drift.schemas import ComplianceRecord, EvaluatorType, InstructionRecord

PY_TAGS = {"python", "py", "python3"}


@dataclass(frozen=True)
class CodeBlock:
    lang: str
    code: str


@dataclass(frozen=True)
class EvalResult:
    compliance: bool | None  # None: the response gives nothing to evaluate
    evidence: str
    failure_type: str | None = None


def code_blocks(text: str) -> list[CodeBlock]:
    """Fenced ``` blocks. An unclosed final fence (truncated output) runs to end of text."""
    blocks = []
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        header, _, body = parts[i].partition("\n")
        blocks.append(CodeBlock(lang=header.strip().lower(), code=body))
    return blocks


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _python_blocks(text: str) -> list[CodeBlock]:
    return [b for b in code_blocks(text) if b.lang in PY_TAGS or (not b.lang and _parses(b.code))]


def json_valid(text: str, params: dict) -> EvalResult:
    s = text.strip()
    if params.get("allow_fence", False):
        blocks = code_blocks(s)
        if len(blocks) == 1 and s.startswith("```") and s.endswith("```"):
            s = blocks[0].code.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        return EvalResult(False, f"JSON parse error: {e.msg} at pos {e.pos}", "invalid_json")
    if params.get("require_object", True) and not isinstance(obj, dict):
        return EvalResult(False, f"Top-level JSON is {type(obj).__name__}, not object",
                          "json_not_object")
    missing = [k for k in params.get("required_keys", []) if k not in obj]
    if missing:
        return EvalResult(False, f"Missing keys: {missing}", "json_missing_keys")
    return EvalResult(True, "Valid JSON object")


def python_only(text: str, params: dict) -> EvalResult:
    blocks = code_blocks(text)
    if not blocks:
        if params.get("require_code", True):
            return EvalResult(False, "No code block in response", "no_code")
        return EvalResult(None, "No code block in response")
    for b in blocks:
        if b.lang in PY_TAGS:
            continue
        if b.lang:
            return EvalResult(False, f"Code block tagged '{b.lang}'", "non_python_code")
        if not _parses(b.code):
            return EvalResult(False, "Untagged code block is not valid Python", "non_python_code")
    return EvalResult(True, f"{len(blocks)} code block(s), all Python")


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)[\w.]*\s+import|import\s+([^#\n]+))", re.M)


def _imported_modules(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        mods = set()
        for frm, imp in _IMPORT_RE.findall(code):
            if frm:
                mods.add(frm)
            else:
                mods.update(p.strip().split()[0].split(".")[0] for p in imp.split(",") if p.strip())
        return mods
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def stdlib_only_imports(text: str, params: dict) -> EvalResult:
    blocks = _python_blocks(text)
    if not blocks:
        return EvalResult(None, "No Python code to check")
    allowed = set(sys.stdlib_module_names) | set(params.get("extra_allowed", []))
    mods = set().union(*(_imported_modules(b.code) for b in blocks))
    bad = sorted(m for m in mods if m not in allowed)
    if bad:
        return EvalResult(False, f"Third-party imports: {bad}", "third_party_import")
    return EvalResult(True, f"Imports all stdlib: {sorted(mods)}")


def max_words(text: str, params: dict) -> EvalResult:
    n = len(re.findall(r"\S+", text))
    limit = int(params["max_words"])
    if n > limit:
        return EvalResult(False, f"{n} words > limit {limit}", "too_long")
    return EvalResult(True, f"{n} words <= limit {limit}")


def starts_with_marker(text: str, params: dict) -> EvalResult:
    marker = params["marker"]
    # Tolerate markdown decoration before the marker (e.g. "**Assumptions:**").
    if re.match(r"^[\s*_#>]*" + re.escape(marker), text, re.IGNORECASE):
        return EvalResult(True, f"Starts with '{marker}'")
    return EvalResult(False, f"Does not start with '{marker}': {text.strip()[:40]!r}",
                      "missing_marker")


def executable_python(text: str, params: dict) -> EvalResult:
    blocks = [b for b in code_blocks(text) if b.lang in PY_TAGS or not b.lang]
    if not blocks:
        return EvalResult(False, "No Python code block", "no_code")
    if any(_parses(b.code) for b in blocks):
        return EvalResult(True, "At least one Python block parses")
    return EvalResult(False, "No Python block parses", "syntax_error")


def preserve_function_names(text: str, params: dict) -> EvalResult:
    required = set(params["function_names"])
    found: set[str] = set()
    for b in code_blocks(text):
        try:
            found |= {n.name for n in ast.walk(ast.parse(b.code))
                      if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
        except SyntaxError:
            found |= set(re.findall(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", b.code, re.M))
    missing = sorted(required - found)
    if missing:
        return EvalResult(False, f"Missing functions: {missing}", "renamed_or_missing_function")
    return EvalResult(True, f"All required functions present: {sorted(required)}")


@dataclass(frozen=True)
class Evaluator:
    name: str
    version: str
    fn: Callable[[str, dict], EvalResult]


EVALUATORS = {
    e.name: e
    for e in [
        Evaluator("json_valid", "1.0", json_valid),
        Evaluator("python_only", "1.0", python_only),
        Evaluator("stdlib_only_imports", "1.0", stdlib_only_imports),
        Evaluator("max_words", "1.0", max_words),
        Evaluator("starts_with_marker", "1.0", starts_with_marker),
        Evaluator("executable_python", "1.0", executable_python),
        Evaluator("preserve_function_names", "1.0", preserve_function_names),
    ]
}


def evaluate(
    instruction: InstructionRecord,
    response_text: str,
    *,
    applicable: bool,
    run_id: str,
    conversation_id: str,
    checkpoint_id: str,
    response_id: str,
    policy_name: str,
) -> ComplianceRecord:
    v = instruction.verification
    ev = EVALUATORS[v.evaluator_name]
    if ev.version != v.evaluator_version:
        raise ValueError(f"{instruction.instruction_id} expects {v.evaluator_name} "
                         f"v{v.evaluator_version}, have v{ev.version}")
    if applicable:
        res = ev.fn(response_text, v.params)
    else:
        res = EvalResult(None, "Instruction not applicable at this checkpoint")
    return ComplianceRecord(
        run_id=run_id,
        conversation_id=conversation_id,
        checkpoint_id=checkpoint_id,
        instruction_id=instruction.instruction_id,
        response_id=response_id,
        policy_name=policy_name,
        applicable=applicable and res.compliance is not None,
        compliance=None if res.compliance is None else int(res.compliance),
        severity=None if res.compliance is None else float(not res.compliance),
        failure_type=res.failure_type,
        evaluator_type=EvaluatorType.DETERMINISTIC,
        evaluator_name=ev.name,
        evaluator_version=ev.version,
        evidence=res.evidence,
    )
