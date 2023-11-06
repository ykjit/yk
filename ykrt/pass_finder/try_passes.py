#!/usr/bin/env python3
"""
usage: python3 try_passes.py <-lto -prelink] -- path path/to/test/oracle

This script aims to identify an LLVM AOT pipeline that satisfies a test oracle.
A typical test oracle would be a script that builds an interpreter and runs a
test suite. This script is required because some passes don't yet work with Yk.

The user specifies at least one pipeline (i.e. -prelink or -lto). The algorithm
then starts with the default clang -O2 pipeline which is sent to the test
oracle for evaluation. Each time the oracle is invoked it is passed a list of
prelink and postlink passes in the environment variable (PRELINK_PASSES and
POSTLINK_PASSES). The oracle must ensure that these pipelines are used when
compiling the interpreter (usually by passing the pipeline to the
--pre-link-pipeline and --post-link-pipeline arguments of yk-config). If the
oracle accepts the pipeline (it returns zero), then the passes are appended to
an "accept list". If the oracle rejects the pipeline (it returns non-zero),
then the algorithm divides the failing pipeline into two parts and tests the
two halves independently (in parallel). The search continues recursively until
the search space is exhausted. Upon termination the final accept list is
printed.
"""
import argparse, io, os, random, shutil, subprocess, sys, time, queue 
from dataclasses import dataclass
from multiprocessing import Manager, Process, Queue, Value
from subprocess import PIPE, Popen

# Stages in an LTO pipeline where optimisation passes can happen.
STAGES = "pre_link", "link_time"
@dataclass
class Pass:
    name: str
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

def get_pipeline_config(is_prelink, try_passes, ok_passes=[]):
    """Returns pipeline configuration based on the flag type."""
    
    if not is_prelink:
        return PipelineConfig([], ok_passes + try_passes)
    else:
        return PipelineConfig(ok_passes + try_passes, [])

def get_opt_cmd(is_prelink):
    """Returns command based on the flag type."""
    
    if not is_prelink:
        return "opt -passes='lto<O2>' -print-pipeline-passes < /dev/null 2>/dev/null"
    else:
        return "opt -passes='lto-pre-link<O2>' -print-pipeline-passes < /dev/null 2>/dev/null" 

def log(logf, s):
    logf.write(s)
    logf.flush()

def executable_exists(cmd):
    return shutil.which(cmd) is not None

def split_passes(passes_string):
    parts = []           
    temp_part = []      
    paren_stack = []     
    angle_stack = []    

    i = 0  
    while i < len(passes_string):
        char = passes_string[i]

        if char == '(':
            paren_stack.append(char)
        elif char == ')':
            if paren_stack:
                paren_stack.pop()

        if char == '<':
            angle_stack.append(char)
        elif char == '>':
            if angle_stack:
                angle_stack.pop()

        if char == ',' and not paren_stack and not angle_stack:
            part = ''.join(temp_part).strip()
            if part:  
                parts.append(part)
            temp_part = []  
        else: 
            temp_part.append(char)

        i += 1  
    
    part = ''.join(temp_part).strip()
    if part:  
        parts.append(part)

    return parts

def get_all_passes(is_prelink):
    #cmd = get_opt_cmd(is_prelink)
    cmd = "opt -passes='lto-pre-link<O2>' -print-pipeline-passes < /dev/null 2>/dev/null"
    sout = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    sout = sout.stdout.decode('utf-8')
    pass_descrs = split_passes(sout)
    print(f"\033[91m{pass_descrs}\033[0m")
    passes = []
    seen = set()

    for part in pass_descrs:
        print(f"\033[92m{part}\033[0m")

    for descr in pass_descrs:
        if descr not in seen:
            seen.add(descr)
            if descr != "BitcodeWriterPass":
                passes.append(Pass(descr))
      
    print(f"Found {len(passes)} passes")    
    return passes

def test_pipeline(logf, pl):
    sys.stdout.write(str(pl) + "...")
    sys.stdout.flush()

    log(logf, "\n\n" + str(pl) + "\n")

    # Make sure we don't run empty strings in pipeline.
    assert (len(pl.pre_link) != 0 or len(pl.link_time) != 0), "Both prelink and postlink passes cannot be empty!!!"

    env = os.environ.copy()  # Create a copy of the environment
    env["PRELINK_PASSES"] = ",".join([p.name for p in pl.pre_link])
    env["LINKTIME_PASSES"] = ",".join([p.name for p in pl.link_time])
    
    print(f"\033[91m!!!!!!\033[0m")
    
    p = subprocess.Popen("try_repeat 1 sh run_tests.sh 2>&1", cwd=CWD, shell=True,
              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    sout, serr =  p.communicate()
    print(f"\033[91m@@@@@@@@@@\033[0m")
    val = sout.strip().split('\n')[-1]
    print(f"\033[91m{val}\033[0m")

    if p.returncode == 0:
        print(" [OK]")
        log(logf, str(pl) + ": OK\n")
    else:
        log(logf, str(pl) + " : FAILED\n")
        print(" [FAIL]")
    return p.returncode == 0, val

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

def evaluate_fitness(logf, is_prelink, entity, passes):
    try_passes = []
    for i, bit in enumerate(entity):
        if bit == 1:
            try_passes.append(passes[i])

    config = get_pipeline_config(is_prelink, try_passes)
    ret, exec_time = test_pipeline(logf, config)
    nFlags = len(try_passes) 
    if ret:
        try:
            exec_time = float(exec_time)  # Convert exec_time to a float
            # Return True for successful programs and execution time as a negative value
            print(f"\033[92m returned {ret} and execution time {exec_time}\033[0m")
            return  [nFlags, exec_time]
        except ValueError:
            print(f"Error converting exec_time to float: {exec_time}")
            return [0, float('inf')]
    else:
        # Return False for programs that crash, and execution time as a large positive value
        return [0, float('inf')]

def crossover(parent1, parent2):
    # Implement crossover logic (e.g., one-point crossover)
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(entity, mutation_rate):
    # Implement mutation logic (bit flip mutation)
    mutated_entity = []
    for bit in entity:
        if random.random() < mutation_rate:
            mutated_entity.append(1 - bit)  # Flip the bit
        else:
            mutated_entity.append(bit)
    return mutated_entity

def genetic_algorithm(logf, is_prelink, population_size, mutation_rate, generations, target_fitness, passes):
    #config = get_pipeline_config(is_prelink, ok_passes, try_passes)
    population = []
    fitness_scores = []

    for _ in range(population_size):
        entity = [random.randint(0,1) for _ in range(len(passes))]
        population.append(entity)
    
    # fitness_scores = [evaluate_fitness(logf, is_prelink, entity, passes) for entity in population]
    # print(f"\033[38;5;128m {fitness_scores}\033[0m")
    for generation in range(generations):
        fitness_scores.clear()
        # Evaluate fitness for each entity in the population
        fitness_scores = [evaluate_fitness(logf, is_prelink, entity, passes) for entity in population]
        # TODO: consider execution time for hyperparameter tuning
        wt = [t[0] for t in fitness_scores] # Choosing weight on basis of OK passes len's 
        print(f"\033[38;5;128m {fitness_scores}\033[0m")
        # Check if we have reached the target fitness
        if target_fitness in fitness_scores:
            print(f"Target fitness reached in generation {generation + 1}!")
            break

        # Select parents for reproduction (roulette wheel selection)
        parents = []
        for _ in range(population_size // 2):
            parent1 = random.choices(population, weights=wt, k=1)[0]
            parent2 = random.choices(population, weights=wt, k=1)[0]
            parents.append((parent1, parent2))

        # Perform crossover and mutation to create a new generation
        new_population = []
        for parent1, parent2 in parents:
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        # Replace the old population with the new generation
        population = new_population

    # Return the best entity in the final population
    wt = [t[0] for t in fitness_scores] 
    best_entity = population[wt.index(max(wt))]
    print(f"\033[35;5;128; m{best_entity}\033[0m")
    return best_entity  

def main(logf, is_prelink):
    #sanity check, test script should work with no extra passes.
    # assert(test_pipeline(logf, PipelineConfig([], [])))

    passes = get_all_passes(is_prelink)
    target_fitness = len(passes)
    best_entity = genetic_algorithm(logf,
        is_prelink,
        population_size = 10, #len(passes) * 2,
        mutation_rate = 0.1,
        generations = 1,
        target_fitness = target_fitness,
        passes = passes,
    )
    final_passes = []
    for i, bit in enumerate(best_entity):
        if bit == 1:
            final_passes.append(passes[i])
    print(f"\033[38;5;128m {final_passes}\033[0m")

if __name__ == "__main__":
    if not os.environ.get('YK_PATH') or not os.environ.get('YKLUA_PATH'):
        print("Please set both YK_PATH and YKLUA_PATH environment variables before running the script.")
        exit(1)
        
    if '--help' in sys.argv:
        print(__doc__)
        exit(0)

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
