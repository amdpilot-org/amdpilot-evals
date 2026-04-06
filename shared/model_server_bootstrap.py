#!/usr/bin/env python3
"""model_server_bootstrap v1 — Lifecycle protocol for model server management.

Replaces inline start_command with explicit lifecycle actions that are used by
both the agent (inner loop: edit → restart → benchmark) and the orchestrator
(outer loop: independent verify).

YAML schema (in task.yaml):
```yaml
model_server:
  version: v1
  lifecycle:
    start:
      command: "python -m sglang.launch_server --model ... --tp 8 --port 30000"
      timeout: 300
      background: true
    healthcheck:
      url: "http://localhost:30000/health"
      interval: 10
      timeout: 5
      retries: 30
    warmup:
      command: "curl -s http://localhost:30000/v1/completions -d '{...}'"
      timeout: 120
    benchmark:
      command: "python benchmark_serving.py ..."
      metric_pattern: 'throughput:\\s+([\\d.]+)'
      runs: 3
      aggregate: median  # or: min, mean
    stop:
      command: "pkill -f sglang.launch_server"
      timeout: 30
  restart_policy:
    file_triggers:                              # globs: any match → must restart
      - "python/sglang/srt/layers/moe/**"
      - "python/sglang/srt/configs/**"
    env_change: restart                         # restart | ignore
    hot_reload: false                           # if true, try SIGHUP before full restart
  kernel_benchmark:                             # alternative to full server for kernel-level evals
    enabled: true
    script: "/workspace/test_harness.py"
    metric_pattern: 'SCORE:\\s+([\\d.]+)'
```

Usage in test harness or orchestrator:
```python
from model_server_bootstrap import ServerLifecycle

lifecycle = ServerLifecycle.from_yaml("task.yaml")
lifecycle.start()
lifecycle.wait_healthy()
lifecycle.warmup()
result = lifecycle.benchmark()
lifecycle.stop()
```
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


@dataclass
class LifecycleAction:
    command: str = ""
    timeout: int = 300
    background: bool = False


@dataclass
class HealthcheckConfig:
    url: str = ""
    command: str = ""
    interval: int = 10
    timeout: int = 5
    retries: int = 30


@dataclass
class BenchmarkAction:
    command: str = ""
    metric_pattern: str = ""
    runs: int = 3
    aggregate: str = "median"  # median | min | mean
    timeout: int = 600


@dataclass
class RestartPolicy:
    file_triggers: list[str] = field(default_factory=list)
    env_change: str = "restart"  # restart | ignore
    hot_reload: bool = False


@dataclass
class ServerConfig:
    version: str = "v1"
    start: LifecycleAction = field(default_factory=LifecycleAction)
    healthcheck: HealthcheckConfig = field(default_factory=HealthcheckConfig)
    warmup: LifecycleAction = field(default_factory=LifecycleAction)
    benchmark: BenchmarkAction = field(default_factory=BenchmarkAction)
    stop: LifecycleAction = field(default_factory=LifecycleAction)
    restart_policy: RestartPolicy = field(default_factory=RestartPolicy)


class ServerLifecycle:
    """Manages model server lifecycle with explicit start/stop/benchmark actions."""

    def __init__(self, config: ServerConfig, workdir: str = "/workspace",
                 exec_fn=None):
        self.config = config
        self.workdir = workdir
        self._exec = exec_fn or self._local_exec
        self._server_proc = None
        self._last_file_state: dict[str, float] = {}

    @classmethod
    def from_yaml(cls, yaml_path: str, **kwargs) -> ServerLifecycle:
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        server_raw = raw.get("model_server", {})
        return cls(cls._parse_config(server_raw), **kwargs)

    @classmethod
    def from_dict(cls, raw: dict, **kwargs) -> ServerLifecycle:
        return cls(cls._parse_config(raw), **kwargs)

    @staticmethod
    def _parse_config(raw: dict) -> ServerConfig:
        lc = raw.get("lifecycle", {})

        start_raw = lc.get("start", {})
        hc_raw = lc.get("healthcheck", {})
        warmup_raw = lc.get("warmup", {})
        bench_raw = lc.get("benchmark", {})
        stop_raw = lc.get("stop", {})
        rp_raw = raw.get("restart_policy", {})

        return ServerConfig(
            version=raw.get("version", "v1"),
            start=LifecycleAction(
                command=start_raw.get("command", ""),
                timeout=start_raw.get("timeout", 300),
                background=start_raw.get("background", True),
            ),
            healthcheck=HealthcheckConfig(
                url=hc_raw.get("url", ""),
                command=hc_raw.get("command", ""),
                interval=hc_raw.get("interval", 10),
                timeout=hc_raw.get("timeout", 5),
                retries=hc_raw.get("retries", 30),
            ),
            warmup=LifecycleAction(
                command=warmup_raw.get("command", ""),
                timeout=warmup_raw.get("timeout", 120),
            ),
            benchmark=BenchmarkAction(
                command=bench_raw.get("command", ""),
                metric_pattern=bench_raw.get("metric_pattern", ""),
                runs=bench_raw.get("runs", 3),
                aggregate=bench_raw.get("aggregate", "median"),
                timeout=bench_raw.get("timeout", 600),
            ),
            stop=LifecycleAction(
                command=stop_raw.get("command", ""),
                timeout=stop_raw.get("timeout", 30),
            ),
            restart_policy=RestartPolicy(
                file_triggers=rp_raw.get("file_triggers", []),
                env_change=rp_raw.get("env_change", "restart"),
                hot_reload=rp_raw.get("hot_reload", False),
            ),
        )

    def _local_exec(self, command: str, timeout: int = 300,
                    background: bool = False) -> subprocess.CompletedProcess:
        if background:
            proc = subprocess.Popen(
                command, shell=True, cwd=self.workdir,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self._server_proc = proc
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.run(
            command, shell=True, cwd=self.workdir,
            capture_output=True, text=True, timeout=timeout,
        )

    def start(self) -> bool:
        if not self.config.start.command:
            return True
        print(f"[bootstrap] Starting server: {self.config.start.command[:80]}...")
        result = self._exec(
            self.config.start.command,
            timeout=self.config.start.timeout,
            background=self.config.start.background,
        )
        self._snapshot_files()
        return result.returncode == 0

    def wait_healthy(self) -> bool:
        hc = self.config.healthcheck
        if not hc.url and not hc.command:
            return True

        print(f"[bootstrap] Waiting for healthy (max {hc.retries} checks)...")
        for i in range(hc.retries):
            try:
                if hc.url:
                    import urllib.request
                    req = urllib.request.Request(hc.url, method="GET")
                    with urllib.request.urlopen(req, timeout=hc.timeout) as resp:
                        if resp.status == 200:
                            print(f"[bootstrap] Healthy after {i+1} checks")
                            return True
                elif hc.command:
                    result = self._exec(hc.command, timeout=hc.timeout)
                    if result.returncode == 0:
                        print(f"[bootstrap] Healthy after {i+1} checks")
                        return True
            except Exception:
                pass
            time.sleep(hc.interval)

        print("[bootstrap] Server failed to become healthy")
        return False

    def warmup(self) -> bool:
        if not self.config.warmup.command:
            return True
        print("[bootstrap] Running warmup...")
        result = self._exec(
            self.config.warmup.command,
            timeout=self.config.warmup.timeout,
        )
        return result.returncode == 0

    def benchmark(self) -> dict[str, Any]:
        """Run benchmark and return {metric: float, raw_outputs: list}."""
        bc = self.config.benchmark
        if not bc.command:
            return {"metric": None, "raw_outputs": []}

        pattern = re.compile(bc.metric_pattern) if bc.metric_pattern else None
        values = []
        raw_outputs = []

        for run_idx in range(bc.runs):
            result = self._exec(bc.command, timeout=bc.timeout)
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            raw_outputs.append(output)

            if pattern:
                matches = pattern.findall(output)
                if matches:
                    val = matches[-1]
                    if isinstance(val, tuple):
                        val = val[0]
                    try:
                        values.append(float(val))
                    except ValueError:
                        pass

        metric = None
        if values:
            if bc.aggregate == "median":
                values.sort()
                metric = values[len(values) // 2]
            elif bc.aggregate == "min":
                metric = min(values)
            elif bc.aggregate == "mean":
                metric = sum(values) / len(values)
            else:
                metric = values[-1]

        return {"metric": metric, "raw_outputs": raw_outputs, "all_values": values}

    def stop(self) -> bool:
        if self.config.stop.command:
            print("[bootstrap] Stopping server...")
            self._exec(self.config.stop.command, timeout=self.config.stop.timeout)
        if self._server_proc:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None
        return True

    def needs_restart(self) -> bool:
        """Check if any file_triggers have been modified since last start."""
        rp = self.config.restart_policy
        if not rp.file_triggers:
            return False

        for pattern in rp.file_triggers:
            base_dir = Path(self.workdir)
            for path in base_dir.rglob("*"):
                rel = str(path.relative_to(base_dir))
                if fnmatch(rel, pattern):
                    mtime = path.stat().st_mtime
                    old_mtime = self._last_file_state.get(rel, 0)
                    if mtime > old_mtime:
                        print(f"[bootstrap] File changed: {rel}")
                        return True
        return False

    def restart_if_needed(self) -> bool:
        """Restart the server if file triggers indicate changes."""
        if not self.needs_restart():
            return True
        print("[bootstrap] Restarting server due to file changes...")
        self.stop()
        ok = self.start()
        if ok:
            ok = self.wait_healthy()
        return ok

    def _snapshot_files(self):
        """Snapshot mtimes of files matching restart_policy triggers."""
        rp = self.config.restart_policy
        self._last_file_state = {}
        for pattern in rp.file_triggers:
            base_dir = Path(self.workdir)
            for path in base_dir.rglob("*"):
                if path.is_file():
                    rel = str(path.relative_to(base_dir))
                    if fnmatch(rel, pattern):
                        self._last_file_state[rel] = path.stat().st_mtime

    def full_cycle(self) -> dict[str, Any]:
        """Run the complete lifecycle: start → healthcheck → warmup → benchmark → stop."""
        results = {"start": False, "healthy": False, "warmup": False,
                    "benchmark": None, "stop": False}

        results["start"] = self.start()
        if not results["start"]:
            return results

        results["healthy"] = self.wait_healthy()
        if not results["healthy"]:
            self.stop()
            return results

        results["warmup"] = self.warmup()
        results["benchmark"] = self.benchmark()
        results["stop"] = self.stop()
        return results
