# TiVy skill (standalone)

A self-contained Claude Code skill that mines a large set of time series for
shared patterns with TiVy and renders the summary to PNG + JSON so an agent can
look at it. Everything the skill needs is in this folder: the library
(`lib/`), the CLI, pinned requirements, example data and scripts, and a
parameter guide with measured effects.

## Install

1. Copy or unzip this folder to `~/.claude/skills/tivy` (all projects) or
   `<project>/.claude/skills/tivy` (one project).
2. Run `./setup.sh` inside it once. It creates `.venv`, installs the
   requirements and renders a demo to `out/setup_check.png`. Needs Python
   3.10+ and a C compiler (macOS `xcode-select --install`, Debian/Ubuntu
   `build-essential`): the dtaidistance DTW library is compiled from source
   against the pinned numpy, because its PyPI wheels for recent Pythons lack
   the C extension TiVy uses and an isolated build targets the wrong numpy.

Claude Code then offers the skill as `/tivy`; see `SKILL.md` for what the
agent is told. Humans can use the same commands directly.

## Contents

| path | what |
|---|---|
| `SKILL.md` | the skill instructions Claude reads |
| `parameters.md` | how every TiVy parameter behaves, with numbers from a sweep |
| `setup.sh` | one-time environment setup + smoke test |
| `tivy_cli.py` | `run` / `render` / `demo` command line |
| `lib/` | `TiVy.py` and the `TSVis` package (copies of the repo root) |
| `examples/` | `run_synthetic.sh`, `run_stock.sh`, `run_ecg.sh`, `prepare_ecg.py`, `stock.pkl` |
| `sweep/` | sweep script, per-dataset `stats_*.csv`, contact sheets per parameter |
| `sync_from_repo.sh` | maintainers: refresh `lib/` and friends from the repo root |

Source repository: <https://github.com/GromitC/TiVy>. The copies in `lib/`
are refreshed with `sync_from_repo.sh`; do not edit them here.
