import click

from . import __version__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.version_option(__version__, '-V', '--version', message='quark %(version)s')
@click.pass_context
def cli(ctx):
    """vector embeddings for quantum circuits — embed, search and dedupe QASM."""
    if ctx.invoked_subcommand is None:
        from . import _style as ui
        ui.banner()
        click.echo()
        click.echo(ctx.get_help())


@cli.command(help="train a model on synthetic rewrite pairs")
@click.option('--samples', default=1500)
@click.option('--epochs', default=12)
@click.option('--batch', default=32)
@click.option('--qmax', default=4)
@click.option('--lr', default=3e-4)
@click.option('--loss', default='infonce', type=click.Choice(['triplet', 'infonce']))
@click.option('--hard-negatives/--no-hard-negatives', default=True)
@click.option('--out', default='quark.pt')
def train(samples, epochs, batch, qmax, lr, loss, hard_negatives, out):
    from .train import train as do_train
    do_train(samples=samples, epochs=epochs, batch=batch, qmax=qmax, lr=lr, loss=loss,
             hard_negatives=hard_negatives, save_to=out)


@cli.command(name='eval', help="evaluate retrieval against hash / sorted / count baselines")
@click.option('--weights', default=None)
@click.option('--queries', default=100)
@click.option('--lib', default=500)
@click.option('--equiv', default=2)
@click.option('--out', default=None)
def evalcmd(weights, queries, lib, equiv, out):
    from .eval_ import run
    run(weights=weights, n_queries=queries, lib_size=lib, equiv_per_query=equiv, out=out)


@cli.command(help="print a sample (anchor, rewrite) pair and its equivalence")
@click.option('--n', default=3, type=int)
@click.option('--depth', default=10, type=int)
@click.option('--k', default=3, type=int)
@click.option('--seed', default=0, type=int)
def show(n, depth, k, seed):
    from .pairs import make_pair
    from .verify import equiv
    from . import _style as ui
    a, b = make_pair(n, depth, k=k, seed=seed)
    click.echo()
    ui.rule('anchor')
    click.echo(a.draw(output='text'))
    click.echo()
    ui.rule('rewrite · positive')
    click.echo(b.draw(output='text'))
    click.echo()
    eq = equiv(a, b)
    ui.status(eq, 'equivalent' if eq else 'not equivalent', 'up to global phase')
    click.echo()


@cli.command(help="quick end-to-end smoke benchmark (no weights needed)")
@click.option('--circuits', default=50, type=click.IntRange(min=1))
@click.option('--phys', default=8, type=click.IntRange(min=2))
@click.option('--seed', default=0, type=int)
@click.option('--weights', default=None)
def bench(circuits, phys, seed, weights):
    from .eval_ import smoke_bench
    smoke_bench(n_circuits=circuits, qmax=phys, seed=seed, weights=weights)


@cli.command(help="group duplicate circuits in a directory of QASM files")
@click.argument('directory')
@click.option('--weights', default=None)
@click.option('--threshold', default=0.9)
@click.option('--verify/--no-verify', default=False)
@click.option('--out', default=None)
def dedupe(directory, weights, threshold, verify, out):
    from .dedupe import cluster, fmt_report, _walk
    from .enc import default_weights
    from . import _style as ui
    weights = weights or default_weights()
    paths = list(_walk(directory))
    if not paths:
        ui.status(False, f'no .qasm files found in {directory}')
        return
    click.echo()
    ui.rule(f'dedupe · {len(paths)} files · threshold {threshold}')
    res = cluster(paths, weights=weights, threshold=threshold, verify=verify)
    click.echo()
    click.echo(fmt_report(res))
    click.echo()
    if out:
        import json as _json
        with open(out, 'w') as f:
            _json.dump({'groups': res['groups'], 'failed': res['failed']}, f, indent=2)


if __name__ == '__main__':
    cli()
