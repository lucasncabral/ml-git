"""
© Copyright 2022 HP Development Company, L.P.
SPDX-License-Identifier: GPL-2.0-only
"""

import click

def _is_field_present(field):
    return True if field else False

def _request_new_value(guide_message, input_message):
    click.echo(guide_message, nl=True)
    field_value = click.prompt(input_message)
    click.echo('')
    return field_value

def _request_user_confirmation(confimation_message):
    should_continue = click.confirm(confimation_message, default=False, abort=True)
    return should_continue

def _abort_click_execution(context):
    context.exit()

def try_wizard_for_field(context, field, guide_message, input_message):
    if _is_field_present(field):
        return field
    else:
        try:
            new_field = _request_new_value(guide_message, input_message)
        except:
            _abort_click_execution(context)