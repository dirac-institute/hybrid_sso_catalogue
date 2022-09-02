# import numpy as np
# from detection_probability import get_detection_probabilities

# probs, unique_objs = get_detection_probabilities(10)
# np.save("probs_pool_test.npy", probs)

# print("NOICE")

# with Pool(32) as pool:
#         probs = pool.map(partial(probability_from_id, sorted_obs=sorted_obs,
#                                  distances=np.logspace(-1, 1, 50) * u.AU,
#                                  radial_velocities=np.linspace(-100, 100, 20) * u.km / u.s,
#                                  first_visit_times=first_visit_times, full_schedule=full_schedule,
#                                  night_lengths=night_lengths, night_list=night_list,
#                                  detection_window=detection_window, min_nights=min_nights), unique_objs)

import astropy.units as u
import numpy as np
from multiprocessing import Pool
from functools import partial
def mult(x, y, z):
    print(x, y, z)
    return x * y * z
with Pool(5) as p:
    x = p.map(partial(mult, y=7, z=2), [1, 2, 3])
    print(x)