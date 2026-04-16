from __future__ import annotations

import time
from pathlib import Path


class ProfileSeries:
    __slots__ = ("count", "total_ms", "min_ms", "max_ms")

    def __init__(self):
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)

    def summary(self) -> dict[str, float]:
        avg_ms = self.total_ms / self.count if self.count > 0 else 0.0
        min_ms = 0.0 if self.count == 0 else self.min_ms
        max_ms = 0.0 if self.count == 0 else self.max_ms
        return {
            "count": self.count,
            "avg_ms": avg_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "total_ms": self.total_ms,
        }


class StepProfiler:
    """Lightweight step profiler with staged marks and periodic reports."""

    __slots__ = (
        "_enabled",
        "_marks",
        "_last",
        "_step_count",
        "_series",
        "_report_interval",
        "_last_report",
        "_report_path",
    )

    def __init__(self, enabled: bool = False, report_interval: int = 200, report_path: Path | None = None):
        self._enabled = enabled
        self._marks: list[tuple[str, float]] = []
        self._last: float = 0.0
        self._step_count: int = 0
        self._series: dict[str, ProfileSeries] = {}
        self._report_interval = report_interval
        self._last_report: int = 0
        self._report_path = report_path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value

    def begin(self) -> None:
        if not self._enabled:
            return
        self._marks.clear()
        self._last = time.perf_counter()

    def mark(self, label: str) -> None:
        if not self._enabled:
            return
        now = time.perf_counter()
        self._marks.append((label, now - self._last))
        self._last = now

    def finish(self) -> dict[str, float] | None:
        if not self._enabled:
            return None
        result: dict[str, float] = {}
        total = 0.0
        for label, dt in self._marks:
            ms = dt * 1000.0
            result[label] = ms
            total += ms
            self._series.setdefault(label, ProfileSeries()).add(ms)
        result["_total"] = total
        self._series.setdefault("_total", ProfileSeries()).add(total)
        self._step_count += 1
        if self.should_report():
            self.write_report()
        return result

    def should_report(self) -> bool:
        return self._enabled and self._step_count > 0 and (self._step_count - self._last_report) >= self._report_interval

    def build_report(self) -> str:
        steps = max(self._step_count - self._last_report, 1)
        lines = [f"[Extractor Profiler] distribution over {steps} steps:"]
        for label in sorted(self._series.keys()):
            summary = self._series[label].summary()
            lines.append(
                f"{label:30s} count={int(summary['count']):6d} avg={summary['avg_ms']:8.3f} ms "
                f"min={summary['min_ms']:8.3f} ms max={summary['max_ms']:8.3f} ms"
            )
        return "\n".join(lines) + "\n"

    def write_report(self) -> Path | None:
        if not self._enabled or self._report_path is None:
            return None
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        self._report_path.write_text(self.build_report(), encoding="utf-8")
        self._series.clear()
        self._last_report = self._step_count
        return self._report_path
