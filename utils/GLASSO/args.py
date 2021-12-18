import argparse

def init_glasso_common():
    def glasso_common_aux():
        glasso_common_parser = argparse.ArgumentParser(add_help=False)
        glasso_common_parser.add_argument(
            '-N', '--N', type=int,  required=False, default=50, dest='N',
            help='Dimension of covariance matrix.')
        glasso_common_parser.add_argument(
            '-sig', '--sig',type=str, required=False, default=None, dest='sig',
            help='Inverse covariance matrix to use.')
        glasso_common_parser.add_argument(
            '-min_samples', '--min_samples', type=int, required=False, default=3, dest='min_samples',
            help='The minimum number of samples.')
        glasso_common_parser.add_argument(
            '-max_samples', '--max_samples', type=int, required=False, default=3, dest='max_samples',
            help='The maximum number of samples.')
        glasso_common_parser.add_argument(
            '-type', '--type', type=str, choices=["chain","planar","random"], required=False, default='chain', dest='type',
            help='The glasso type of covariance matrix.')
        glasso_common_parser.add_argument(
            '-type_param', '--type_param', type=float, required=False, default=1.0, dest='type_param',
            help='The hyperparamter of the covariance matrix (density for random, max distance for planar).')
        glasso_common_parser.add_argument(
            '-id_add', '--id_add', type=float, required=False, default=1.0, dest='id_add',
            help='The multiplier of Identity matrix which is added to the generated covariance matrix.')
        glasso_common_parser.add_argument(
           '-nr', '--normal', action='store_true', required=False, default=False, dest='normal',
            help='Normalize test set.')
        glasso_common_parser.set_defaults(cuda=False)
        glasso_common_parser.set_defaults(lam=None)
        glasso_common_parser.set_defaults(gista_T=None)
        glasso_common_parser.set_defaults(test_mode=None)
        glasso_common_parser.set_defaults(epsilon_tol=None)

        return glasso_common_parser

    full_glasso_common_parser = glasso_common_aux()
    full_glasso_common_parser.add_argument(
        '-l', '--lam', required=False, type=float, default=0.1, dest='lam',
        help='The regularization paramater.')
    full_glasso_common_parser.add_argument(
        '-gista', '--gista_T', required=False, type=int, default=0, dest='gista_T',
        help='The number of iteration for GISTA result compare.')

    test_glasso_common_parser = glasso_common_aux()
    test_glasso_common_parser.add_argument(
        '-l', '--lam', required=False, type=float, default=0.1, dest='lam',
        help='The regularization paramater.')
    test_glasso_common_parser.add_argument(
        '-gista', '--gista_T', required=False, type=int, default=0, dest='gista_T',
        help='The number of iteration for GISTA result compare.')
    test_glasso_common_parser.add_argument(
        '-cr', '--criteria', type=str, required=False, choices=["Gap", "Rel", "Nmse", "Diff"], default="Gap", dest='test_mode',
        help="The stop criteria to use.")
    test_glasso_common_parser.add_argument(
        '-eps', '--epsilon_tol', required=False, type=float, default=1e-2, dest='epsilon_tol',
        help='The epsilon tolerance for stop criteria.')

    generate_glasso_common_parser = glasso_common_aux()

    return full_glasso_common_parser, test_glasso_common_parser, generate_glasso_common_parser

def glasso_full_test_generate_parser(
        full_subparser, test_subparser, generate_subparser,
        glasso_full_parser, glasso_test_parser, glasso_generate_parser,
        full_common_parser, test_common_parser, generate_common_parser
    ):
    #Full + Test
    from algos.GLASSO.GISTA import init_GISTA_parser
    init_GISTA_parser(full_subparser.add_parser('GISTA', parents=[full_common_parser,glasso_full_parser]))
    init_GISTA_parser(test_subparser.add_parser('GISTA', parents=[test_common_parser,glasso_test_parser]))

    from algos.GLASSO.QUIC import init_QUIC_parser
    init_QUIC_parser(full_subparser.add_parser('QUIC', parents=[full_common_parser,glasso_full_parser]))
    init_QUIC_parser(test_subparser.add_parser('QUIC', parents=[test_common_parser,glasso_test_parser]))

    from algos.GLASSO.ALM import init_ALM_parser
    init_ALM_parser(full_subparser.add_parser('ALM', parents=[full_common_parser,glasso_full_parser]))
    init_ALM_parser(test_subparser.add_parser('ALM', parents=[test_common_parser,glasso_test_parser]))

    from algos.GLASSO.OBN import init_OBN_parser
    init_OBN_parser(full_subparser.add_parser('OBN', parents=[full_common_parser,glasso_full_parser]))
    init_OBN_parser(test_subparser.add_parser('OBN', parents=[test_common_parser,glasso_test_parser]))

    from algos.GLASSO.NL_fista import init_NL_fista_parser
    init_NL_fista_parser(full_subparser.add_parser('NL_fista', parents=[full_common_parser,glasso_full_parser]))
    init_NL_fista_parser(test_subparser.add_parser('NL_fista', parents=[test_common_parser,glasso_test_parser]))

    from algos.GLASSO.pISTA import init_pISTA_parser
    init_pISTA_parser(full_subparser.add_parser('pISTA', parents=[full_common_parser,glasso_full_parser]))
    init_pISTA_parser(test_subparser.add_parser('pISTA', parents=[test_common_parser,glasso_test_parser]))

    #Data generation
    generate_data_parser = generate_subparser.add_parser('data', help='Generate data.', parents=[generate_common_parser, glasso_generate_parser])
    generate_data_parser.set_defaults(object='data')
