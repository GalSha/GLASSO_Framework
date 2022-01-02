def setup_algo(args):
    algo = None

    if args.algo == 'GISTA':
        # GISTA
        if args.cuda:
            from algos.GLASSO.cuda_GISTA import cuda_GISTA as GISTA
        else:
            from algos.GLASSO.GISTA import GISTA
        algo = GISTA(T=args.T, N=args.N, lam=args.lam, ls_iter=args.ls_iter, step_lim=args.step_lim)

    if args.algo == 'QUIC':
        # QUIC
        if args.cuda:
            from algos.GLASSO.cuda_QUIC import cuda_QUIC as QUIC
        else:
            from algos.GLASSO.QUIC import QUIC
        algo = QUIC(T=args.T, N=args.N, lam=args.lam, inner_T=args.inner_T, armijo_iter=args.armijo_iter,
                    step_lim=args.step_lim)

    if args.algo == 'ALM':
        # ALM
        if args.cuda:
            from algos.GLASSO.cuda_ALM import cuda_ALM as ALM
        else:
            from algos.GLASSO.ALM import ALM
        algo = ALM(T=args.T, N=args.N, lam=args.lam, N_mu=args.N_mu,
                   eta=args.eta, skip=args.skip, step_lim=args.step_lim)

    if args.algo == 'OBN':
        # OBN
        if args.cuda:
            from algos.GLASSO.cuda_OBN import cuda_OBN as OBN
        else:
            from algos.GLASSO.OBN import OBN
        algo = OBN(T=args.T, N=args.N, lam=args.lam, inner_T=args.inner_T,
                        ls_iter=args.ls_iter, step_lim=args.step_lim)

    if args.algo == 'NL_fista':
        # NL_fista
        if args.cuda:
            from algos.GLASSO.cuda_NL_fista import cuda_NL_fista as NL_fista
        else:
            from algos.GLASSO.NL_fista import NL_fista
        algo = NL_fista(T=args.T, N=args.N, lam=args.lam, inner_T=args.inner_T,
                        ls_iter=args.ls_iter, step_lim=args.step_lim)

    if args.algo == 'pISTA':
        # pISTA
        if args.cuda:
            from algos.GLASSO.cuda_pISTA import cuda_pISTA as pISTA
        else:
            from algos.GLASSO.pISTA import pISTA
        algo = pISTA(T=args.T, N=args.N, lam=args.lam, ls_iter=args.ls_iter, step_lim=args.step_lim,
                     init_step=args.init_step, hybrid=args.hybrid)

    return algo