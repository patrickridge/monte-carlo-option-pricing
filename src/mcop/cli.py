# src/mcop/cli.py

import argparse

from mcop.simulate_paths import simulate_gbm_paths
from mcop.american_lsm import american_option_lsm


def cmd_price(args: argparse.Namespace) -> None:
    """
    Price an American option using LSM (Python or C++).
    Simulates GBM paths then runs LSM backward induction.
    """
    paths = simulate_gbm_paths(
        S0=args.S0,
        r=args.r,
        sigma=args.sigma,
        T=args.T,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        q=args.q,
        seed=args.seed,
        antithetic=True,
    )

    if args.engine == "cpp":
        # Lazy import so Python engine works even if the C++ extension isn't built
        try:
            from mcop.american_lsm_cpp import american_option_lsm_cpp
        except Exception as e:
            raise SystemExit(
                "C++ engine requested but C++ extension is not available.\n"
                "Build it first (see README), or run with --engine python."
            ) from e

        price = american_option_lsm_cpp(
            paths,
            K=args.K,
            r=args.r,
            T=args.T,
            is_call=args.call,
            degree=args.degree,
        )
    else:
        price = american_option_lsm(
            paths,
            K=args.K,
            r=args.r,
            T=args.T,
            is_call=args.call,
            degree=args.degree,
            q=args.q,
        )

    opt_type = "call" if args.call else "put"
    print(
        f"American {opt_type} price (LSM, {args.engine}): {price:.6f} "
        f"[S0={args.S0}, K={args.K}, T={args.T}, r={args.r}, q={args.q}, sigma={args.sigma}, "
        f"steps={args.n_steps}, paths={args.n_paths}, degree={args.degree}]"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcop",
        description="Monte Carlo option pricing tools",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- price command ----
    p_price = sub.add_parser(
        "price",
        help="Price an American option using LSM",
    )

    p_price.add_argument("--S0", type=float, default=100.0)
    p_price.add_argument("--K", type=float, default=100.0)
    p_price.add_argument("--r", type=float, default=0.05)
    p_price.add_argument("--q", type=float, default=0.0)
    p_price.add_argument("--sigma", type=float, default=0.2)
    p_price.add_argument("--T", type=float, default=1.0)

    p_price.add_argument("--n-steps", dest="n_steps", type=int, default=100)
    p_price.add_argument("--n-paths", dest="n_paths", type=int, default=50_000)
    p_price.add_argument("--degree", type=int, default=2)
    p_price.add_argument("--seed", type=int, default=123)

    p_price.add_argument(
        "--call",
        action="store_true",
        help="Price a call (default: put)",
    )

    p_price.add_argument(
        "--engine",
        choices=["python", "cpp"],
        default="python",
        help="Pricing engine to use",
    )

    p_price.set_defaults(func=cmd_price)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()