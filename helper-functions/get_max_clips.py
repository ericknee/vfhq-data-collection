import math
# ['0:3', '60:3', '300:3', '600:3', '1200:3', '1800:4', '2400:5', '3000:6', '3600:6', '4200:7', '4800:8', '5400:10']
def get_max_clips(video_duration, min_value=3, max_value=10):
    min_duration = 60         # 1 minute
    max_duration = 5400       # 90 minutes
    clamped_duration = max(min_duration, min(max_duration, video_duration))
    exp_duration = math.exp(clamped_duration / 5400)
    normalized_exp_duration = (exp_duration - math.exp(min_duration / 5400)) / (math.exp(max_duration / 5400) - math.exp(min_duration / 5400))
    result = min_value + normalized_exp_duration * (max_value - min_value)
    return int(result)

times = [0, 60, 300, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400]
results = [f"{time}:{get_max_clips(time)}" for time in times]
print(results)