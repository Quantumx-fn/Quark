import click

# one cyan accent over dim-gray scaffolding — kept deliberately spare.
ACCENT = 'cyan'
DIM = 'bright_black'
OK = 'green'
WARN = 'yellow'
BAD = 'red'


def paint(text, fg=None, bold=False, dim=False):
    return click.style(text, fg=fg, bold=bold, dim=dim)


def rule(label='', width=62):
    if label:
        head = f'── {label} '
        click.echo('  ' + paint(head + '─' * max(4, width - len(head)), fg=DIM))
    else:
        click.echo('  ' + paint('─' * width, fg=DIM))


def banner():
    click.echo()
    click.echo('  ' + paint('◢◤ quark', fg=ACCENT, bold=True)
               + '   ' + paint('vector embeddings for quantum circuits', fg=DIM))


def kv(label, value, unit=''):
    line = '  ' + paint(f'{label:<15}', fg=DIM) + paint(str(value), bold=True)
    if unit:
        line += ' ' + paint(unit, fg=DIM)
    click.echo(line)


def status(ok, text, note=''):
    mark = paint('✓ ', fg=OK) if ok else paint('✗ ', fg=BAD)
    line = '  ' + mark + paint(text, bold=True)
    if note:
        line += paint('  ' + note, fg=DIM)
    click.echo(line)


def _best(text, target):
    try:
        return target is not None and abs(float(text) - target) < 1e-12
    except ValueError:
        return False


def table(headers, rows, highlight=None):
    # headers: column titles; rows: list of cells (col 0 = label, the rest numeric
    # strings). per-column winners and the highlighted row are drawn in the accent.
    cols = len(headers)
    w = [len(headers[j]) for j in range(cols)]
    for row in rows:
        for j in range(cols):
            w[j] = max(w[j], len(row[j]))

    best = {}
    for j in range(1, cols):
        nums = []
        for row in rows:
            try:
                nums.append(float(row[j]))
            except ValueError:
                pass
        best[j] = max(nums) if nums else None

    def line(l, mid, r):
        return '  ' + paint(l + mid.join('─' * (w[j] + 2) for j in range(cols)) + r, fg=DIM)

    def emit(cells):
        bar = paint('│', fg=DIM)
        click.echo('  ' + bar + bar.join(' ' + cells[j] + ' ' for j in range(cols)) + bar)

    def cell(text, j, *, header=False, hl=False, win=False):
        pad = text.ljust(w[j]) if j == 0 else text.rjust(w[j])
        if header or win:
            return paint(pad, fg=ACCENT, bold=True)
        if hl:
            return paint(pad, bold=True)
        return pad

    click.echo(line('┌', '┬', '┐'))
    emit([cell(headers[j], j, header=True) for j in range(cols)])
    click.echo(line('├', '┼', '┤'))
    for row in rows:
        hl = highlight is not None and row[0] == highlight
        emit([cell(row[j], j, hl=hl, win=(j >= 1 and _best(row[j], best[j]))) for j in range(cols)])
    click.echo(line('└', '┴', '┘'))
