FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326

ARG MERGE_COMMIT=385a35bd118d2e420be1fd6978c703d22a3bcc77

# Clone the sglang repo and check out to the parent of the merge commit
RUN git clone https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && \
    git checkout ${MERGE_COMMIT}~1 && \
    pip install --no-deps -e python/ && \
    # Clean git history to save space
    rm -rf .git

WORKDIR /workspace/sglang

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
