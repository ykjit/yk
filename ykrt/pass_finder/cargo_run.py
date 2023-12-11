import os, subprocess, time

def run_test(yk_path,env=None, n=1) :
    if env is None:
        env = os.environ.copy()
    env['PATH'] = f"{yk_path}/bin:{env['PATH']}"
    env['YK_BUILD_TYPE'] = 'debug'
    postlink_passes = (env or os.environ).get('POSTLINK_PASSES', 'Not Set')
    print(f"\033[93mPOSTLINK_PASSES: {postlink_passes}\033[0m")
    os.chdir(yk_path)
    r = subprocess.run(["timeout 50 cargo test"], shell=True, env=env or os.environ)
    # assert r.returncode == 0
    if r.returncode == 0:
        os.chdir("/home/shreei/research/yklua")
        times = []
        c_test = None
        for _ in range(n):
            subprocess.run(["make clean"], shell=True, env=env)
            c = subprocess.run(["make && timeout 2 sh test.sh"], shell=True, env=env or os.environ)
            # assert c.returncode == 0
            if c.returncode == 0:
                before = time.time()
                c_test = subprocess.run(["timeout 2 sh run.sh"], shell=True, env=env or os.environ)
                elapsed = time.time() - before
                times.append(elapsed)

        # assert c_test.returncode == 0
        assert len(times) == 0
        if len(times) != 0:
            mean_time = sum(times) / len(times)
        else: 
            mean_time = 0
        os.chdir(yk_path)
        return c_test.returncode if (not c_test == None) else c.returncode, mean_time
    os.chdir(yk_path)
    return r.returncode, 0 
    # print(mean_time)
    # print(f"returncode: {c.returncode}") 

code, time_p = run_test("/home/shreei/research/yk_pv")
print(code, " ",  time_p)

