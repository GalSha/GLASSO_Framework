def setup_task_problem(args):
    if args.problem == 'glasso':
        # GLASSO
        from tasks.GLASSO import GLASSO
        from algos.GLASSO.setup import setup_algo
        algo = setup_algo(args)
        problem = GLASSO(algo=algo, N=args.N, sig=args.sig, min_samples=args.min_samples, max_samples=args.max_samples,
                         type=args.type, type_param=args.type_param, id_add=args.id_add, normal=args.normal, lam=args.lam,
                         test_mode=args.test_mode, epsilon_tol=args.epsilon_tol, gista_T=args.gista_T, cuda=args.cuda)
    return problem