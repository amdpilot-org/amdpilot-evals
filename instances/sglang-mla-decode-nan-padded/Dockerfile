FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
WORKDIR /workspace

# Clone sglang and checkout pre-fix state (before PR #19902)
RUN rm -rf /sgl-workspace/sglang && \
    git clone --depth 1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git fetch --depth 100 origin 27053aa5ed2d7e8f080720ec1ad96cbeb353bb59 && \
    git checkout -f 27053aa5ed2d7e8f080720ec1ad96cbeb353bb59~1 && \
    /opt/venv/bin/pip install --no-deps -e python/ && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

# Fix circular import in aiter: fused_fp8_quant.py accesses aiter.dtypes.fp8
# at module level before aiter.__init__ finishes. Defer to lazy property.
COPY fix_aiter_import.py /tmp/fix_aiter_import.py
RUN /opt/venv/bin/python3 /tmp/fix_aiter_import.py

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/sglang /workspace/sglang && \
    ln -sf /sgl-workspace/aiter /workspace/aiter 2>/dev/null; true

CMD ["sleep", "infinity"]
