import subprocess
import time
import sys, os

def run_test(yk_path,env=None, n=1) :
    prelink_passes = (env or os.environ).get('PRELINK_PASSES', 'Not Set')
    print(f"\033[93mPRELINK_PASSES: {prelink_passes}\033[0m")
    os.chdir("/home/shreei/research/yk_pv")
    times = []
    for i in range(n):
        before = time.time() 
        try:
            c = subprocess.run(["timeout 60 cargo test"], shell=True, env=env or os.environ) 
        except:
            print(f"\033[92m @@@@@@@@@@@@@@\033[0m")
        elapsed = time.time() - before
        times.append(elapsed)
    mean_time = sum(times) / len(times)
    # print(mean_time)
    # print(f"returncode: {c.returncode}")
    return c.returncode, mean_time 
# run_test("/home/shreei/research/yk_pv")
