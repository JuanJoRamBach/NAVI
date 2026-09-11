"""
jobs/eval_gate.py

Stage 0's missing half (NAVI reliability plan — see IDEAS.md's
"Dispatcher reliability" section for the full staged plan and why this
exists). storage/usage.py answers "what did NAVI's routing cost"; this
file answers "was a routing choice actually GOOD," which nothing
answered before now — model_ranking.py's own snapshot explicitly,
deliberately never auto-applies to config/store.py's routing because
there was nothing yet to judge a candidate against beyond "it used fewer
tokens."

Runs a small, FIXED, repeatable task suite against a candidate
(provider, model) rather than live production traffic — the same
methodology real routing benchmarks use (Artificial Analysis's Intelligence
Index, RouterArena): score against a held-constant task set, not raw
usage, since NAVI's real traffic has no fixed ground truth to check
against. Every EVAL_TASKS entry uses a DETERMINISTIC check (a tool call
happened, a keyword is present, no error prefix) rather than an LLM-judge
rubric — preferring a checkable mechanism over an opaque LLM judgment is
the same standing bias the rest of this reliability thread has followed
(see how_to_handle_context.md's own "prefer a checkable artifact over an
opaque AI synthesis" philosophy).

Metric is TOKENS per successful task, not dollars. NAVI's real routing
candidates (Groq, Cloudflare free tier, etc.) are free/near-free — a $
comparison between them is meaningless, the same reasoning
storage/usage.py's get_savings_summary() already used to justify NOT
attempting a real $-spent figure. The dollar-denominated savings claim
(vs. reference flagship models like Fable 5.1/Astra) is a completely
separate, brochure-facing metric this file never touches.

Real API calls, not simulated — running this costs real quota against
whichever candidate you point it at, same as any other NAVI request.

NOT wired into anything automatic. Stage 2 (auto-apply model_ranking.py's
snapshot, gated by this) is still explicitly out of scope — this file
only makes that gate POSSIBLE to build, it doesn't call it from anywhere
yet. Run it by hand: `python -m jobs.eval_gate <provider> <model>`.
"""

import argparse
from dataclasses import dataclass, field
from typing import Callable

from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_provider
from tools.registry import schemas_for


def _check_no_error(text: str | None, tool_names: list[str]) -> bool:
    return bool(text) and not text.strip().startswith("⚠️")


def _check_contains(*keywords: str) -> Callable[[str | None, list[str]], bool]:
    def check(text: str | None, tool_names: list[str]) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(k.lower() in lowered for k in keywords)
    return check


def _check_tool_called(tool_name: str) -> Callable[[str | None, list[str]], bool]:
    def check(text: str | None, tool_names: list[str]) -> bool:
        return tool_name in tool_names
    return check


@dataclass
class EvalTask:
    id: str
    system_prompt: str
    user_prompt: str
    tools: list[str] = field(default_factory=list)
    check: Callable[[str | None, list[str]], bool] = _check_no_error


# Deliberately small and narrow for a first real version — 5 tasks
# spanning NAVI's real command/mode surface (plain chat, a checkable
# fact, tool use, Agent Work's create_workflow shape, strict instruction-
# following), not an exhaustive benchmark. Grow this over time with real
# failure cases as they're found, same "start with the dumbest version
# that could work" discipline the rest of Stage 0 followed.
EVAL_TASKS: list[EvalTask] = [
    EvalTask(
        id="idle_chat_basic",
        system_prompt="You are NAVI, a helpful assistant. Answer directly and concisely.",
        user_prompt="What can you help me with? Answer in one sentence.",
        check=_check_no_error,
    ),
    EvalTask(
        id="arithmetic_fact",
        system_prompt="You are NAVI, a helpful assistant. Answer directly and concisely.",
        user_prompt="What is 15% of 240? Reply with just the number.",
        check=_check_contains("36"),
    ),
    EvalTask(
        id="tool_use_web_search",
        system_prompt="You are NAVI's research assistant. Use the web_search tool when you need current information you don't already know.",
        user_prompt="Search the web for NAVI's own GitHub repository and tell me one fact about it.",
        tools=["web_search"],
        check=_check_tool_called("web_search"),
    ),
    EvalTask(
        id="workflow_creation_shape",
        system_prompt="You are Agent Work Chat. When asked to build a workflow, call create_workflow with a real, complete steps list — don't just describe it in prose.",
        user_prompt="Create a one-step workflow named 'Test Workflow' whose only step outputs the text 'hello'.",
        tools=["create_workflow"],
        check=_check_tool_called("create_workflow"),
    ),
    EvalTask(
        id="instruction_following_format",
        system_prompt="You are NAVI, a helpful assistant. Follow formatting instructions exactly.",
        user_prompt="Reply with exactly the single word: acknowledged",
        check=_check_contains("acknowledged"),
    ),
]


def run_eval_task(task: EvalTask, provider_name: str, model: str) -> dict:
    """Runs ONE fixed eval task against a real (provider, model)
    candidate. Returns {task_id, passed, prompt_tokens,
    completion_tokens, error}. A ProviderError or missing config counts
    as a fail, not a skip — a candidate that can't be reached is not a
    viable candidate."""
    try:
        provider = get_provider(provider_name)
    except ProviderNotConfigured as e:
        return {"task_id": task.id, "passed": False, "prompt_tokens": 0, "completion_tokens": 0, "error": str(e)}

    tools = schemas_for(task.tools) if task.tools else None
    messages = [
        ChatMessage(role="system", content=task.system_prompt),
        ChatMessage(role="user", content=task.user_prompt),
    ]
    try:
        response = provider.chat(model=model, messages=messages, tools=tools)
    except ProviderError as e:
        return {"task_id": task.id, "passed": False, "prompt_tokens": 0, "completion_tokens": 0, "error": str(e)}

    tool_names = [tc.name for tc in response.tool_calls]
    passed = task.check(response.text, tool_names)
    usage = (response.raw or {}).get("usage") or {}
    return {
        "task_id": task.id, "passed": passed,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "error": None,
    }


def run_eval_suite(provider_name: str, model: str) -> dict:
    """Runs the whole fixed EVAL_TASKS suite against one (provider,
    model) candidate — real API calls, costs real quota. Returns
    {provider, model, results, tasks, passed, pass_rate,
    total_prompt_tokens, total_completion_tokens, tokens_per_success}.
    tokens_per_success is None when nothing passed — a candidate that
    fails everything has no defined cost-per-success, not zero."""
    results = [run_eval_task(t, provider_name, model) for t in EVAL_TASKS]
    passed = sum(1 for r in results if r["passed"])
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    total_tokens = total_prompt + total_completion
    return {
        "provider": provider_name, "model": model, "results": results,
        "tasks": len(EVAL_TASKS), "passed": passed,
        "pass_rate": (passed / len(EVAL_TASKS)) if EVAL_TASKS else 0.0,
        "total_prompt_tokens": total_prompt, "total_completion_tokens": total_completion,
        "tokens_per_success": (total_tokens / passed) if passed else None,
    }


def should_promote(baseline: dict, candidate: dict) -> tuple[bool, str]:
    """The actual gate Stage 2 will eventually call before ever writing a
    candidate into config/store.py's routing. Deliberately conservative —
    no reliability regression tolerated, matching this whole reliability
    thread's standing bias toward checkable, not-overclaimed wins. A
    candidate must hold or improve pass_rate AND strictly reduce
    tokens_per_success to be promotable; anything else is a reject, not a
    maybe."""
    if candidate["pass_rate"] < baseline["pass_rate"]:
        return False, f"pass rate regressed: {candidate['pass_rate']:.0%} vs baseline {baseline['pass_rate']:.0%}"
    if candidate["passed"] == 0:
        return False, "candidate passed zero eval tasks"
    if baseline["passed"] == 0:
        return True, "baseline passed zero tasks — any working candidate is an improvement"
    if candidate["tokens_per_success"] >= baseline["tokens_per_success"]:
        return False, (
            f"no token-efficiency gain: {candidate['tokens_per_success']:.0f} vs "
            f"baseline {baseline['tokens_per_success']:.0f} tokens/success"
        )
    return True, (
        f"holds pass rate ({candidate['pass_rate']:.0%}) and reduces tokens/success "
        f"({baseline['tokens_per_success']:.0f} -> {candidate['tokens_per_success']:.0f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NAVI's fixed eval-gate task suite against a real candidate.")
    parser.add_argument("provider")
    parser.add_argument("model")
    parser.add_argument("--baseline-provider", default=None)
    parser.add_argument("--baseline-model", default=None)
    args = parser.parse_args()

    candidate = run_eval_suite(args.provider, args.model)
    print(f"\nCandidate: {candidate['provider']}/{candidate['model']}")
    for r in candidate["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        note = f" ({r['error']})" if r["error"] else ""
        print(f"  [{status}] {r['task_id']}{note}")
    print(f"  {candidate['passed']}/{candidate['tasks']} passed ({candidate['pass_rate']:.0%})")
    tps = candidate["tokens_per_success"]
    print(f"  tokens/success: {tps:.0f}" if tps is not None else "  tokens/success: n/a (nothing passed)")

    if args.baseline_provider and args.baseline_model:
        baseline = run_eval_suite(args.baseline_provider, args.baseline_model)
        print(f"\nBaseline: {baseline['provider']}/{baseline['model']}")
        print(f"  {baseline['passed']}/{baseline['tasks']} passed ({baseline['pass_rate']:.0%})")
        btps = baseline["tokens_per_success"]
        print(f"  tokens/success: {btps:.0f}" if btps is not None else "  tokens/success: n/a (nothing passed)")

        promote, reason = should_promote(baseline, candidate)
        verdict = "PROMOTE" if promote else "REJECT"
        print(f"\n{verdict}: {reason}")


if __name__ == "__main__":
    main()
