# Eval Instance Harness Checklist

Before launching any experiment, verify every item. Derived from vllm-run-lessons.md (lessons 1-20).

## Dockerfile

- [ ] **Base image correct** — `sgl-dev` for SGLang/aiter, `vllm-dev` for vLLM
- [ ] **Commit pinned correctly** — for regression bugs: after bug-introducing PR, before fix PR (lesson #16)
- [ ] **Commit is merge_commit~1** — the fix does NOT exist in the checkout
- [ ] **[vLLM only] Wheel removed + editable install** — `pip uninstall -y vllm && pip install -e . --no-deps --no-build-isolation`
- [ ] **[vLLM only] PYTHONPATH set** — `ENV PYTHONPATH=/workspace/vllm`
- [ ] **Safe kill script installed** — `/usr/local/bin/safe-kill-server` uses `pgrep -f 'python3 -m (sglang|vllm)' | xargs -r kill -9`

## task_description.md

- [ ] **Python path matches image family** — SGLang: `/opt/venv/bin/python3`, vLLM: `/usr/bin/python3`
- [ ] **No `unset PYTHONPATH`** — if PYTHONPATH needed, say `export PYTHONPATH=...`
- [ ] **Kill pattern uses safe template** — `pgrep -f 'python3 -m sglang' | xargs -r kill -9` (not `kill -9 $(pgrep -f sglang)`)
- [ ] **No solution code leaked** — describes symptom only, not the fix. Check for exact fix patterns like `layer.k_scale if layer.k_scale is not None else self.k_scale` or function names like `fused_gemm_afp4wfp4_split_cat`
- [ ] **Affected files listed** — but no diff or patch content
- [ ] **Clear "do NOT modify test harness" instruction**

## test_harness.py

- [ ] **kill_server() uses safe pattern** — `pgrep -f 'python3 -m sglang' | xargs -r kill -9`, NOT `kill -9 $(pgrep -f sglang)`
- [ ] **Python path correct** — `PYTHON = "/opt/venv/bin/python3"` or `"/usr/bin/python3"`
- [ ] **No `env.pop("PYTHONPATH")` or `env["PYTHONPATH"] = ""`** in vLLM harnesses
- [ ] **Server flags match issue reproduction** — no extra flags like `--disable-cuda-graph` that bypass crash path (lesson #19)
- [ ] **Test model triggers bug path** — tiny/2-layer models may not exercise the buggy codepath (lesson #18)
- [ ] **No broad `except Exception: pass`** — catch-all handlers mask the target bug. Only catch specific exceptions (lesson: fused-moe-fix harness scored 100 on buggy code because `except Exception` caught everything)
- [ ] **Harness strong enough** — unmodified baseline fails reliably (test with preflight gate)
- [ ] **Harness reachable** — a perfect fix can score 100% (lesson #10)
- [ ] **SCORE output format** — prints `SCORE: <0-100>` (matches metric_pattern in task.yaml)
- [ ] **Sufficient test volume** — enough prompts/requests that flaky pass rate < 1%
- [ ] **Timeout adequate** — model loading time accounted for (600s+ for large models)

## task.yaml

- [ ] **benchmark.command uses correct Python path**
- [ ] **metric_pattern matches harness output** — `'SCORE:\s+([\d.]+)'`
- [ ] **base_image matches Dockerfile FROM**
- [ ] **name field matches experiment_id** (for DB sync)

## Pre-launch

- [ ] **Manual reproduction verification** — run harness in container manually, confirm bug reproduces (score < 100). Do NOT rely on "issue is open" as evidence (lessons #17, #20)
- [ ] **Preflight baseline gate** — run harness on unmodified image, confirm score < 100. This is the last defense, not the only one
- [ ] **Docker build succeeds** — `docker build -t eval-name .`
- [ ] **Harness runs without errors** — test in container before launching agent
- [ ] **Bug is in the right repo** — verify fix belongs to the workspace repo, not a dependency like aiter (lesson #11)
