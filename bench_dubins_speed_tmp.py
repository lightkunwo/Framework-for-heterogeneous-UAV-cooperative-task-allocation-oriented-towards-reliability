import time, math, random
import dubins

pairs = []
for i in range(2000):
    s = [random.uniform(0,8000), random.uniform(0,5000), random.uniform(-math.pi, math.pi)]
    e = [random.uniform(0,8000), random.uniform(0,5000), random.uniform(-math.pi, math.pi)]
    pairs.append((s,e,50.0))

t0 = time.perf_counter()
for s,e,r in pairs:
    dubins.shortest_path(s,e,r).path_length()
dt = time.perf_counter() - t0
print('dubins_file=', dubins.__file__)
print('calls=', len(pairs))
print('seconds=', dt)
print('per_call_ms=', dt/len(pairs)*1000)
print('estimate_2995200_calls_seconds=', dt/len(pairs)*2995200)
print('estimate_minutes=', dt/len(pairs)*2995200/60)
