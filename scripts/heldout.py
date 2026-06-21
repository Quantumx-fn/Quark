import argparse
import json
from quark.train import train
from quark.eval_ import run
from quark.pairs import REWRITES, REWRITE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', default='insert_id')
    ap.add_argument('--samples', type=int, default=1500)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--loss', default='infonce')
    ap.add_argument('--out', default='heldout.json')
    args = ap.parse_args()

    train_rw = [r for r in REWRITES if r.__name__ != args.holdout]
    test_rw = [REWRITE_NAMES[args.holdout]]

    print(f"\n==== held-out: {args.holdout} ====")
    print(f"train rewrites: {[r.__name__ for r in train_rw]}")
    print(f"eval rewrites:  {[r.__name__ for r in test_rw]}")
    print()

    print(">>> training without held-out rewrite")
    train(samples=args.samples, epochs=args.epochs, loss=args.loss,
          rewrites=train_rw, hard_negatives=True, save_to=f"heldout_{args.holdout}.pt")

    print()
    print(">>> eval on training-distribution (sanity)")
    in_dist = run(weights=f"heldout_{args.holdout}.pt", n_queries=100, lib_size=500,
                  equiv_per_query=2, rewrites=train_rw)

    print()
    print(">>> eval on held-out rewrite ONLY")
    out_dist = run(weights=f"heldout_{args.holdout}.pt", n_queries=100, lib_size=500,
                   equiv_per_query=2, rewrites=test_rw)

    delta_r1 = out_dist['quark_r@1'] - in_dist['quark_r@1']
    delta_r10 = out_dist['quark_r@10'] - in_dist['quark_r@10']
    print()
    print(f"  Δ R@1  : {delta_r1:+.3f}  (out_dist - in_dist)")
    print(f"  Δ R@10 : {delta_r10:+.3f}")
    if out_dist['quark_r@10'] >= 0.6 * in_dist['quark_r@10']:
        verdict = "PASS — model generalizes to unseen rewrite"
    elif out_dist['quark_r@10'] >= 0.3 * in_dist['quark_r@10']:
        verdict = "PARTIAL — degraded but still beats baselines"
    else:
        verdict = "FAIL — model is rewrite-pattern-matching, not learning structure"
    print()
    print(f"  verdict: {verdict}")

    with open(args.out, 'w') as f:
        json.dump({'holdout': args.holdout, 'in_dist': in_dist, 'out_dist': out_dist, 'verdict': verdict}, f, indent=2)

    return out_dist


if __name__ == '__main__':
    main()
