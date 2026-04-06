# Multimodal API: corrupt / truncated images yield server errors

When a client sends a **base64-encoded image** that is **truncated or otherwise corrupt**, the server responds with an **HTTP 500** internal error instead of treating it as bad user input (**HTTP 400**). The failure happens while decoding or loading the image bytes (for example during PIL decode), and the error is not mapped to a client-facing validation error.

## Affected area

- Image loading for multimodal / vision requests (base64 image payloads)

## Environment

- vLLM source at `/workspace/vllm`
- Use `/opt/venv/bin/python3` for checks

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
