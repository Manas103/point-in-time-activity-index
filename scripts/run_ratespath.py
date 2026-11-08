"""Entry point: run the meeting-dated policy path and macro-surprise study
and print every measured claim. Mirrors scripts/run_measurements.py's shape."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ratespath import measurements


def main():
    print("=== Reprice tolerance (16-meeting path vs. 11 OIS curve pillars) ===")
    print(json.dumps(measurements.measure_reprice_tolerance(), indent=2, default=str))

    print("\n=== Release count and join ===")
    print(json.dumps(measurements.measure_release_count(), indent=2, default=str))

    print("\n=== Event study: policy-path sensitivity to macro surprises ===")
    print(json.dumps(measurements.measure_event_study(), indent=2, default=str))


if __name__ == "__main__":
    main()
