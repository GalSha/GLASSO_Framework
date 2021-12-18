import argparse

def init_full_common():
    ###   Full common args   ###
    full_common_parser = argparse.ArgumentParser(add_help=False)
    full_common_parser.add_argument(
        '-s', '--seed', type=int, required=False, default=None, dest='seed',
        help='Constant seed for random numbers (Default: random).')
    full_common_parser.add_argument(
        '-id', '--exp_id', type=int, required=False, default=0, dest='exp_id', help='Experiment id number.')
    full_common_parser.add_argument(
        '-tbs', '--tbs', type=int, required=False, default=10, dest='tbs', help='Test batch size.')
    full_common_parser.add_argument(
        '-db', '--db', type=str, required=False, default=None, dest='db',
        help='Pre generated database directory.')
    full_common_parser.add_argument(
        '-H', '--save_history', required=False, action='store_true',default=False,dest='hist',
        help='Save the solution history.')
    full_common_parser.add_argument(
        '-cuda', '--cuda', action='store_true', required=False, default=False, dest='cuda',
        help='use cuda with cupy library.')
    return full_common_parser

def init_test_common():
    ###   Test common args   ###
    test_common_patser = argparse.ArgumentParser(add_help=False)
    test_common_patser.add_argument(
        '-s', '--seed', type=int, required=False, default=None, dest='seed',
        help='Constant seed for random numbers (Default: random).')
    test_common_patser.add_argument(
        '-id', '--exp_id', type=int, required=False, default=0, dest='exp_id', help='Experiment id number.')
    test_common_patser.add_argument(
        '-tbs', '--tbs', type=int, required=False, default=10, dest='tbs', help='Test batch size.')
    test_common_patser.add_argument(
        '-db', '--db', type=str, required=False, default=None, dest='db',
        help='Pre generated database directory.')
    test_common_patser.add_argument(
        '-H', '--save_history', required=False, action='store_true',default=False,dest='hist',
        help='Save the solution history.')
    test_common_patser.add_argument(
        '-no_timer', '--no_timer', required=False, action='store_false',default=True,dest='timer',
        help='Do not include stop criteria computation in the time measurements.')
    test_common_patser.add_argument(
        '-cuda', '--cuda', action='store_true', required=False, default=False, dest='cuda',
        help='use cuda with cupy library.')
    return test_common_patser

def init_generate_common():
    ###   Generate common args   ###
    generate_common_patser = argparse.ArgumentParser(add_help=False)
    generate_common_patser.add_argument(
        '-s', '--seed', type=int, required=False, default=None, dest='seed',
        help='Constant seed for random numbers (Default: random).')
    generate_common_patser.add_argument(
        '-id', '--id', type=int, required=False, default=0, dest='id', help='Data generation id number.')
    generate_common_patser.add_argument(
        '-tbs', '--tbs', type=int, required=False, default=10, dest='tbs', help='Batch size to generate.')
    generate_common_patser.add_argument(
        '-cuda', '--cuda', action='store_true', required=False, default=False, dest='cuda',
        help='use cuda with cupy library.')
    return generate_common_patser

def init_full_test_generate_parser(problem_parser):
    subparsers = problem_parser.add_subparsers(title='Task', description='Possible tasks.')

    # Test Op
    full_parser = subparsers.add_parser('full', help="Per iteration status of the problem")
    full_parser.set_defaults(task='full')
    full_subparser = full_parser.add_subparsers(title='Algorithms', description='Possible algorithms.')

    # End Op
    test_parser = subparsers.add_parser('test', help="Run the problem until stop criteria")
    test_parser.set_defaults(task='test')
    test_subparser = test_parser.add_subparsers(title='Algorithms', description='Possible algorithms.')

    # Data generation
    generate_parser = subparsers.add_parser('generate', help="Generate problem data")
    generate_parser.set_defaults(task='generate')
    generate_subparser = generate_parser.add_subparsers(title='Objects', description='Possible objects to generate.')

    return full_subparser, test_subparser, generate_subparser

def init_parser():
    full_common_parser = init_full_common()
    test_common_parser = init_test_common()
    generate_common_parser = init_generate_common()

    parser = argparse.ArgumentParser(description='General numrical framework.')
    parser.set_defaults(problem=None)
    parser.set_defaults(task=None)
    parser.set_defaults(algo=None)
    parser.set_defaults(object=None)
    subparsers = parser.add_subparsers(title='Problems', description='Possible problems.')


    # Graphical Lasso
    from utils.GLASSO.args import init_glasso_common, glasso_full_test_generate_parser
    glasso_parser = subparsers.add_parser('glasso', help='Graphical LASSO problem.')
    glasso_parser.set_defaults(problem='glasso')
    glasso_parser.set_defaults(problem_parser=glasso_parser)
    full_glasso_common_parser, test_glasso_common_parser, generate_glasso_commno_parser = init_glasso_common()
    full_subparser, test_subparser, generate_subparser = init_full_test_generate_parser(glasso_parser)
    glasso_full_test_generate_parser(
        full_subparser, test_subparser, generate_subparser,
        full_glasso_common_parser, test_glasso_common_parser, generate_glasso_commno_parser,
        full_common_parser, test_common_parser, generate_common_parser
    )

    return parser

def get_args():
    parser = init_parser()
    args = parser.parse_args()

    if args.problem is None:
        parser.parse_args(['-h'])
        exit(-1)
    elif args.task is None:
        parser.parse_args([args.problem,'-h'])
        exit(-1)
    elif args.object is None and args.algo is None:
        parser.parse_args([args.problem,args.task,'-h'])
        exit(-1)


    return args