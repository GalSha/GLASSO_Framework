import os
import numpy as np
from tasks.setup import setup_task_problem

def run_task(args):
    if args.task == 'generate': run_generate(args)
    elif args.task == 'full': run_full(args)
    elif args.task == 'test': run_test(args)

def run_full(args):
    problem = setup_task_problem(args)
    problem.init_full(args.seed, args.tbs, args.db, args.hist)
    problem.full()

    print("Init   | {full_status}"\
          .format(full_status=problem.full_status(0)))
    for t in range(args.T):
        print("T={t:4} | {full_status}"\
          .format(t=t+1,full_status=problem.full_status(t + 1)))

    np.savez('./results/{problem}/{problem_name}_exp{exp_id}_full_result'
                .format(problem=args.problem.upper(), problem_name=problem.name(),exp_id=args.exp_id), **problem.full_result())
    if args.hist:
        np.save('./results/{problem}/{problem_name}_exp{exp_id}_full_hist'
                .format(problem=args.problem.upper(), problem_name=problem.name(),exp_id=args.exp_id),problem.full_hist())

    return None

def run_test(args):
    problem = setup_task_problem(args)
    problem.init_test(args.seed, args.tbs, args.db, args.hist, args.timer)
    problem.test()

    print("Average    | {test_status}"\
      .format(test_status=problem.test_status(0)))

    for t in range(args.tbs):
        print("Index={t:4} | {test_status}"\
          .format(t=t+1,test_status=problem.test_status(t+1)))

    np.savez('./results/{problem}/{problem_name}_exp{exp_id}_test_result'
                .format(problem=args.problem.upper(), problem_name=problem.name(),exp_id=args.exp_id), **problem.test_result())
    if args.hist:
        np.save('./results/{problem}/{problem_name}_exp{exp_id}_test_hist'
                .format(problem=args.problem.upper(), problem_name=problem.name(),exp_id=args.exp_id),problem.test_hist())

    return None

def run_generate(args):
    print("Generating Data:")
    print("Number of data to generate: {tbs}".format(tbs=args.tbs))
    problem = setup_task_problem(args)
    problem.init_generate(args.object, args.seed, args.tbs)
    problem.generate()

    print("Average    | {generate_status}"\
      .format(generate_status=problem.generate_status(0)))

    for t in range(args.tbs):
        print("Index={t:4} | {generate_status}"\
          .format(t=t+1,generate_status=problem.generate_status(t+1)))

    generate_name = problem.generate_name()

    path = "./db/{problem}/{generate_name}_id{id}".format(problem=args.problem.upper(), generate_name=generate_name, id=args.id)
    if not os.path.exists(path):
        os.mkdir(path)

    problem.generate_save(path)

    print("Done!")
    return None