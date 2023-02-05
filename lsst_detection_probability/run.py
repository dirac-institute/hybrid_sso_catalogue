from detection_probability import get_detection_probabilities
import time

run_start = time.time()

for i in range(291, 365):
    print(f"\nStarting night {i}")
    start = time.time()
    get_detection_probabilities(night_start=i, obj_type="mba")
    print(f"Time for this run: {time.time() - start:1.1f}s")

print(f"\n\nOverall, it took {time.time() - run_start:1.1f}s")
