import os

prof = profiler.Profiler(
    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
    scheduler=(1, 9),
    on_trace_ready=profiler.export_chrome_tracing('./log'))
prof.start()
for iter in range(10):
    #train()
    prof.step()
