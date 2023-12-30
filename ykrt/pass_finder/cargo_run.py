import os
import subprocess
import time

def run_test(yk_path, env=None, n=1):
    if env is None:
        env = os.environ.copy()
    env['PATH'] = f"{yk_path}/bin:{env['PATH']}"
    env['YK_BUILD_TYPE'] = 'debug'
    postlink_passes = (env or os.environ).get('POSTLINK_PASSES', 'Not Set')
    print(f"\033[93mPOSTLINK_PASSES: {postlink_passes}\033[0m")
    os.chdir(yk_path)
    # assert r.returncode == 0
    os.chdir("/home/shreei/research/yklua")
    times = []
    c_test = None
    env['IR_CHANGE'] = "false"
    for _ in range(n):
        # if IR changes returncode will be 0
        c = subprocess.run(["timeout 30 sh test.sh"], shell=True,
                           env=env or os.environ)
        print(f"\033[95m returncode for c is {c.returncode} \033[0m\n")
        # assert c.returncode == 0
        os.chdir(yk_path)
        r = subprocess.run(["timeout 50 cargo test"], shell=True,
                           env=env or os.environ)
        if c.returncode == 0 and r.returncode == 0:
            os.chdir("/home/shreei/research/yklua")
            before = time.time()
            c_test = subprocess.run(["timeout 15 sh run.sh"],
                                    shell=True, env=env or os.environ)
            elapsed = time.time() - before
            times.append(elapsed)

    # assert c_test.returncode == 0
    if len(times) != 0:
        mean_time = sum(times) / len(times)
    else:
        mean_time = 0
        print(f"\033[95m mean time: {mean_time} \033[0m\n") 
    os.chdir(yk_path)
    if c_test is None:
        return 130, 0
    
    return c.returncode, mean_time

code, time_p = run_test("/home/shreei/research/yk_pv")
print(code, " ",  time_p)
