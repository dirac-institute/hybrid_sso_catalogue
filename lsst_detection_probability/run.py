from detection_probability import get_detection_probabilities

for i in range(365):
    print(f"\nStarting night {i}")
    get_detection_probabilities(night_start=i, obj_type="neo")