"""
dispatcher/scheduler.py

In-process cron scheduler — no external ping needed (GitHub Actions,
cron-job.org, an OS-level crontab entry on the Lightsail box) for periodic
jobs. NAVI's own running process parses a standard 5-field Unix cron
expression (via croniter) and fires the registered job itself, in-process,
no network hop at all.

JuanJo's call (2026-09-01), after walking through GitHub Actions (real
scheduler drift under load — see dispatcher/reminders.py's own docstring
for the exact incident) and an OS-level Lightsail crontab entry (works,
but needs SSH access to set up and lives outside this repo entirely):
"a python function that saves the cron... and it's fire per the syntax"
instead. One background thread per registered job — simplest correct
approach given this will realistically only ever have a handful of jobs,
not worth a single merged-loop scheduler's added complexity.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from croniter import croniter


@dataclass
class ScheduledJob:
    name: str
    cron_expr: str
    fn: Callable[[], Awaitable[None]]


_jobs: list[ScheduledJob] = []
_started = False


def register_job(name: str, cron_expr: str, fn: Callable[[], Awaitable[None]]) -> None:
    """Registers an async, zero-argument job to run on the given
    standard 5-field cron schedule. Call before start_scheduler() —
    jobs registered after starting never get picked up (no dynamic
    add/remove yet, not needed for the handful of jobs this has today)."""
    _jobs.append(ScheduledJob(name=name, cron_expr=cron_expr, fn=fn))


def start_scheduler() -> None:
    """Spawns one daemon thread per registered job. Idempotent — a
    second call is a no-op, so it's safe to call from a startup hook
    that could in principle run more than once."""
    global _started
    if _started:
        return
    _started = True
    for job in _jobs:
        threading.Thread(target=_run_job_loop, args=(job,), daemon=True).start()


def _run_job_loop(job: ScheduledJob) -> None:
    # Survives a server restart without any explicit persistence in
    # THIS module (JuanJo, 2026-09-01: "remember it needs to survive
    # restarts") — for two separate reasons, not one:
    #   1. The cron expression itself lives in config/store.py, which
    #      is Filen-backed and restored on boot — server.py's startup
    #      hook re-registers this job fresh every time the process
    #      starts, reading whatever cron string is currently saved.
    #   2. croniter here is seeded from the CURRENT time on every
    #      restart, not resumed from wherever it left off — which is
    #      the correct behavior for a recurring check, not a gap: the
    #      real state that matters (each workflow's own
    #      trigger.next_run_at) lives in agent_work.db on Lightsail's
    #      persistent disk, not scratch. If the process was down when a
    #      workflow's next_run_at passed, it's simply still due the
    #      moment this loop's next tick fires after restart —
    #      due_workflows() checks next_run_at <= now, so nothing gets
    #      silently skipped, it just runs a bit late.
    cron = croniter(job.cron_expr, time.time())
    while True:
        next_fire = cron.get_next(float)
        sleep_for = max(0.0, next_fire - time.time())
        time.sleep(sleep_for)
        try:
            asyncio.run(job.fn())
        except Exception as e:
            # A failed job (e.g. a transient DB error) shouldn't kill
            # this thread — the loop just keeps going and tries again
            # at the next scheduled fire, same "don't let one bad tick
            # take down the whole mechanism" reasoning as every other
            # background loop in this codebase.
            print(f"[scheduler] job '{job.name}' failed: {e}")
