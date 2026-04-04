FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/sglang && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin c6850ac30cbea6e751575beb150818e757c710b2 && \
    git checkout -f c6850ac30cbea6e751575beb150818e757c710b2~1 && \
    /opt/venv/bin/pip install --no-deps -e python/
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/sglang /workspace/sglang
CMD ["sleep", "infinity"]
