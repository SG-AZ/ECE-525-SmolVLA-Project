"""checking the real simulator with optional remote inference."""

import argparse
import json

from .app import handle
from .backend import Backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    backend = Backend()
    target = next(iter(backend.targets))
    try:
        for operation in ["reset", "capture"] + (["infer"] if args.infer else []):
            result = handle(backend, operation, 0, target)
            print(
                json.dumps(
                    {
                        "operation": operation,
                        "camera_shapes": [
                            list(image.shape) if image is not None else None for image in result[:2]
                        ],
                        "state": result[2],
                        "task": result[3],
                        "latency_ms": result[-2],
                        "status": result[-1],
                    }
                ),
                flush=True,
            )
            if result[-1].startswith("ERROR"):
                raise SystemExit(1)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
