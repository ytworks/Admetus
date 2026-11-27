"""The ``admetus`` command line.

Flag names are deliberately uniform across subcommands: every one that selects
endpoints takes a repeatable ``--endpoint`` plus ``--all``. Inconsistent names
between subcommands are the usual reason a generated invocation fails.
"""

from __future__ import annotations

import click

from admetus.cli.cmd_data import data_group
from admetus.cli.cmd_doctor import doctor_command
from admetus.cli.cmd_endpoints import endpoints_command
from admetus.cli.cmd_models import models_group
from admetus.cli.cmd_predict import predict_command
from admetus.cli.cmd_train import evaluate_command, train_command
from admetus.version import get_version


@click.group(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100})
@click.version_option(version=get_version(), prog_name="admetus")
@click.option(
    "--log-level", type=click.Choice(["debug", "info", "warning", "error"]), default="info",
    help="Logging threshold. Logs go to standard error.",
)
@click.option(
    "--log-format", type=click.Choice(["text", "json"]), default="text",
    help="Log as prose or as one JSON object per line.",
)
@click.option("--quiet", is_flag=True, help="Only warnings and errors.")
@click.pass_context
def cli(ctx: click.Context, log_level: str, log_format: str, quiet: bool) -> None:
    """Predict ADMET properties from SMILES.

    Models are LightGBM fold ensembles over a Morgan fingerprint concatenated with
    a 2D pharmacophore signature, trained one model per public dataset.

    Start with 'admetus doctor' to check the install, then 'admetus endpoints' to
    see what can be predicted.
    """
    ctx.ensure_object(dict)


cli.add_command(doctor_command)
cli.add_command(endpoints_command)
cli.add_command(data_group)
cli.add_command(train_command)
cli.add_command(evaluate_command)
cli.add_command(predict_command)
cli.add_command(models_group)


def main() -> None:
    """Entry point for the ``admetus`` script."""
    cli(obj={})


__all__ = ["cli", "main"]
