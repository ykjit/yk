#!/usr/bin/env python3

import argparse, math, os, shutil, sys, time, queue 
from dataclasses import dataclass
from multiprocessing import Manager, Process, Queue, Value
from subprocess import PIPE, Popen

# Stages in an LTO pipeline where optimisation passes can happen.
STAGES = "pre_link", "link_time"
@dataclass
class Pass:
    name: str
    # FIXME: implement support for pass parameters?
    #args: list

    def __str__(self):
        return self.name

@dataclass
class PipelineConfig:
    pre_link: list
    link_time: list

    def __str__(self):
        pre_link = ",".join([ str(p) for p in self.pre_link ])
        link_time = ",".join([ str(p) for p in self.link_time ])
        return f"PipelineConfig(pre_link=[{pre_link}], link_time=[{link_time}])"

def get_pipeline_config(is_prelink, ok_passes, try_passes):
    """Returns pipeline configuration based on the flag type."""
    
    if not is_prelink:
        return PipelineConfig([], ok_passes + try_passes)
    else:
        return PipelineConfig(ok_passes + try_passes, [])

def get_opt_cmd(is_prelink):
    """Returns command based on the flag type."""
    
    if not is_prelink:
        return "opt -passes='lto<O2>' -print-pipeline-passes < /dev/null 2>/dev/null | tr ',' '\n'"
    else:
        return "opt -passes='lto-pre-link<O2>' -print-pipeline-passes < /dev/null 2>/dev/null | tr ',' '\n'"

def log(logf, s):
    logf.write(s)
    logf.flush()

def executable_exists(cmd):
    return shutil.which(cmd) is not None

def get_all_passes(is_prelink):
    cmd = get_opt_cmd(is_prelink) 
    p = Popen(cmd, shell=True, stdout=PIPE, close_fds=True)
    sout, serr = p.communicate() 
    assert(p.returncode == 0)

    sout = sout.decode()
    pass_descrs = [x.strip() for x in sout.strip().split("\n")]

    passes = []
    seen = set()
     
    for descr in pass_descrs:
        parts = descr.split("<")
        if parts[0] not in seen:
            seen.add(parts[0])
            passes.append(Pass(parts[0]))

    print(f"Found {len(passes)} passes")    
    return passes

def test_pipeline(logf, pl):
    sys.stdout.write(str(pl) + "...")
    sys.stdout.flush()

    log(logf, "\n\n" + str(pl) + "\n")  

    # Make sure we don't run empty strings in pipeline.
    assert (len(pl.pre_link) != 0 and len(pl.link_time) != 0), "Both prelink and postlink passes cannot be empty!!!"

    env = os.environ
    env["PRELINK_PASSES"] = ",".join([p.name for p in pl.pre_link])
    env["LINKTIME_PASSES"] = ",".join([p.name for p in pl.link_time])

    p = Popen("try_repeat 10 sh test.sh 2>&1", cwd=CWD, shell=True,
              stdout=PIPE, close_fds=True, env=env)
    
    sout, _ =  p.communicate()
    log(logf, sout.decode())

    if p.returncode == 0:
        print(" [OK]")
        log(logf, str(pl) + ": OK\n")
    else:
        log(logf, str(pl) + " : FAILED\n")
        print(" [FAIL]")
    return p.returncode == 0

def list_of_passes_to_str(passes):
    return ",".join([str(p) for p in passes])

def binary_split_worker(logf, ok_passes, passes_tasks, is_prelink, processing):
    while True:
        try:
            # Wait for a task in the queue for worker_timeout seconds
            try_passes = passes_tasks.get(block=True)
            processing.value += 1
        except queue.Empty:
            print(f"\033[91mClosing Worker\033[0m")
            break
        else:
            if try_passes == "CLOSE":
                print(f"\033[33mClosing Worker\033[0m")
                break
            log(logf, f">>> Trying to add:\n{list_of_passes_to_str(try_passes)}\n\n")
            
            # Choose pipeline based on flag type.
            config = get_pipeline_config(is_prelink, ok_passes, try_passes)

            if test_pipeline(logf, config):
                ok_passes.extend(try_passes)
            elif len(try_passes) == 1:
                return
            else:
                subset_len = len(try_passes)
                subset1 = try_passes[:subset_len // 2] 
                subset2 = try_passes[subset_len // 2:]

                # Put the 2 halves of the split in the worker queue to be processed concurrently
                passes_tasks.put(subset1)
                passes_tasks.put(subset2)
            processing.value -= 1
    return True

def binary_split(logf, passes, is_prelink):
    # [config] number of worker processes
    # Increasing this number might increase the processing time due to locks on ok_passes
    # and the worker queue (passes_tasks).
    # The worker-queue pattern solves the thread bomb problem. 
    n_processes = 10
    with Manager() as manager:
        # Shared state list for OK passes
        ok_passes = manager.list()
        # Worker queue for passes to be processed
        passes_tasks = Queue()
        processes = []
        processing = Value("i", 0)

        passes_tasks.put(passes)  # put set of all passes in worker queue

        # Start `n_processes` workers
        for w in range(n_processes):
            p  = Process(target=binary_split_worker, args=(logf, ok_passes, passes_tasks, is_prelink, processing))
            processes.append(p)
            p.start()

        while processing.value > 0:
            time.sleep(5)

        for w in range(n_processes):
            passes_tasks.put("CLOSE")
        
        for p in processes:
            p.join()

        print("\n\nFinal OK passes")
        print(72 * "=")
        print(list_of_passes_to_str(ok_passes))

def main(logf, is_prelink):
    #sanity check, test script should work with no extra passes.
    # assert(test_pipeline(logf, PipelineConfig([], [])))

    passes = get_all_passes(is_prelink)
    binary_split(logf, passes, is_prelink)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-lto', action='store_true', help='Set flag for LTO.')
    group.add_argument('-prelink', action='store_true', help='Set flag for Prelink.')
    parser.add_argument('--path', required=True, help='The path to set as cwd.')

    args = parser.parse_args()

    is_prelink = args.prelink

    if not is_prelink and not args.lto:
        print("Flag invalid! Please provide a valid flag: -lto or -prelink")
        exit(1)

    if not os.path.isdir(args.path):
        print(f"Invalid path: {args.path}. Please provide a valid directory for yklua.")
        exit(1)

    # Set the global variable with the parsed path
    CWD = args.path
    print(f"PATH to interpreter: {CWD}")

    with open("passes.log", "w") as logf:
        main(logf, is_prelink)
