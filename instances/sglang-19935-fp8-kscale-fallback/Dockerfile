FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226

WORKDIR /workspace

# Historical bundle reconstructed from legacy sglang-sft materials.
# Commit-window pinning for this issue still needs a dedicated pass.
RUN ln -sfn /sgl-workspace/sglang /workspace/sglang

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN printf '#!/bin/bash\npgrep -f "python3 -m sglang" | xargs -r kill -9 2>/dev/null\nsleep 3\n' \
    > /usr/local/bin/safe-kill-server && chmod +x /usr/local/bin/safe-kill-server

CMD ["sleep", "infinity"]
