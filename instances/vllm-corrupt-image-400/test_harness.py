#!/usr/bin/env python3
"""Test harness for vllm-corrupt-image-400.

Tests that corrupt or truncated image data is handled gracefully by the
multimodal image loading API.
"""
import subprocess
import sys

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)


def run_py(code: str, timeout: int = 120):
    return subprocess.run(
        ["/opt/venv/bin/python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


print("=" * 60)
print("vllm-corrupt-image-400 test harness")
print("=" * 60)

# Behavioral script adapted from tests/multimodal/media/test_image.py (truncated PNG cases)
SCRIPT = r"""
from io import BytesIO

import pybase64
from PIL import Image

from vllm.multimodal.media import ImageMediaIO


def classify(exc):
    return f"{type(exc).__name__}:{exc!r}"


def try_load_bytes(label, data: bytes):
    io = ImageMediaIO()
    try:
        r = io.load_bytes(data)
    except ValueError as e:
        if "Failed to load image" in str(e):
            print(f"{label}:VALUE_ERROR")
        else:
            print(f"{label}:VALUE_ERROR_OTHER:{classify(e)}")
        return
    except OSError as e:
        print(f"{label}:OSERROR:{classify(e)}")
        return
    except Image.UnidentifiedImageError as e:
        print(f"{label}:UNIDENTIFIED:{classify(e)}")
        return
    except Exception as e:
        print(f"{label}:OTHER:{classify(e)}")
        return
    try:
        r.media.load()
    except OSError as e:
        print(f"{label}:OSERROR_LAZY:{classify(e)}")
        return
    except Exception as e:
        print(f"{label}:OTHER_LAZY:{classify(e)}")
        return
    print(f"{label}:NO_ERROR")


def try_load_base64_truncated_real_png():
    buf = BytesIO()
    img = Image.new("RGB", (8, 8), (100, 150, 200))
    img.save(buf, format="PNG")
    full = buf.getvalue()
    trunc = full[: max(1, len(full) // 2)]
    b64 = pybase64.b64encode(trunc).decode("ascii")
    io = ImageMediaIO()
    label = "TRUNC_B64"
    try:
        r = io.load_base64("image/png", b64)
    except ValueError as e:
        if "Failed to load image" in str(e):
            print(f"{label}:VALUE_ERROR")
        else:
            print(f"{label}:VALUE_ERROR_OTHER:{classify(e)}")
        return
    except OSError as e:
        print(f"{label}:OSERROR:{classify(e)}")
        return
    except Image.UnidentifiedImageError as e:
        print(f"{label}:UNIDENTIFIED:{classify(e)}")
        return
    except Exception as e:
        print(f"{label}:OTHER:{classify(e)}")
        return
    try:
        r.media.load()
    except OSError as e:
        print(f"{label}:OSERROR_LAZY:{classify(e)}")
        return
    except Exception as e:
        print(f"{label}:OTHER_LAZY:{classify(e)}")
        return
    print(f"{label}:NO_ERROR")


# Short bogus PNG signature (from upstream test)
try_load_bytes("HDR_BYTES", b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
try_load_base64_truncated_real_png()
"""

r = run_py(SCRIPT)
combined = (r.stdout or "") + (r.stderr or "")
for line in combined.strip().splitlines():
    print(f"  {line}")

hdr_ok = "HDR_BYTES:VALUE_ERROR" in combined
trunc_ok = "TRUNC_B64:VALUE_ERROR" in combined

check(
    "corrupt PNG header bytes -> ValueError (Failed to load image)",
    hdr_ok,
    combined.strip()[:1500] if not hdr_ok else "",
)
check(
    "truncated real PNG via load_base64 -> ValueError (Failed to load image)",
    trunc_ok,
    combined.strip()[:1500] if not trunc_ok else "",
)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
