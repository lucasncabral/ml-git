"""
© Copyright 2022 HP Development Company, L.P.
SPDX-License-Identifier: GPL-2.0-only
"""

import os
import unittest
from unittest import mock

import pytest

from ml_git.commands.wizard import WizardMode
from ml_git.ml_git_message import output_messages
from tests.integration.commands import MLGIT_REMOTE_FSCK, MLGIT_CHECKOUT, MLGIT_COMMIT, MLGIT_CONFIG_WIZARD, \
    MLGIT_CREATE, MLGIT_INIT, MLGIT_ENTITY_INIT, MLGIT_REMOTE_ADD
from tests.integration.helper import MLGIT_ADD, GLOBAL_CONFIG_PATH, DATASETS, MODELS, GIT_PATH, ERROR_MESSAGE
from tests.integration.helper import check_output


@pytest.mark.usefixtures('tmp_dir')
class WizardCommandsAcceptanceTests(unittest.TestCase):
    def setup_wizard(self):
        mode = WizardMode.ENABLED.value
        self.assertIn(output_messages['INFO_WIZARD_MODE_CHANGED'].format(mode),
                      check_output(MLGIT_CONFIG_WIZARD % mode))

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    @mock.patch.dict(os.environ, {'HOME': GLOBAL_CONFIG_PATH})
    def test_01_dataset_wizard_commands_project_not_initialized(self):
        self.setup_wizard()

        self.assertIn(output_messages['ERROR_NOT_IN_RESPOSITORY'],
                      check_output(MLGIT_CREATE % ('datasets', 'ENTITY-NAME')))

        self.assertIn(output_messages['ERROR_NOT_IN_RESPOSITORY'],
                      check_output(MLGIT_REMOTE_FSCK % ('datasets', 'ENTITY-NAME')))

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    @mock.patch.dict(os.environ, {'HOME': GLOBAL_CONFIG_PATH})
    def test_02_model_wizard_commands_project_not_initialized(self):
        self.setup_wizard()

        self.assertIn(output_messages['ERROR_NOT_IN_RESPOSITORY'],
                      check_output(MLGIT_CHECKOUT % ('models', 'ENTITY-NAME') + ' --wizard'))

        self.assertIn(output_messages['ERROR_NOT_IN_RESPOSITORY'],
                      check_output(MLGIT_ADD % ('models', 'ENTITY-NAME', '--wizard')))

        self.assertIn(output_messages['ERROR_NOT_IN_RESPOSITORY'],
                      check_output(MLGIT_COMMIT % ('models', 'ENTITY-NAME', '--wizard')))

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    @mock.patch.dict(os.environ, {'HOME': GLOBAL_CONFIG_PATH})
    def test_03_wizard_commands_entity_not_initialized(self):
        self.setup_wizard()
        self.assertIn(output_messages['INFO_INITIALIZED_PROJECT_IN'] % self.tmp_dir, check_output(MLGIT_INIT))

        self.assertIn(output_messages['ERROR_NOT_INITIALIZED'] % DATASETS,
                      check_output(MLGIT_CREATE % ('datasets', 'ENTITY-NAME') + ' --wizard'))

        self.assertIn(output_messages['ERROR_NOT_INITIALIZED'] % DATASETS,
                      check_output(MLGIT_REMOTE_FSCK % ('datasets', 'ENTITY-NAME') + ' --wizard'))

        self.assertIn(output_messages['ERROR_NOT_INITIALIZED'] % MODELS,
                      check_output(MLGIT_CHECKOUT % ('models', 'ENTITY-NAME') + ' --wizard'))

        self.assertIn(output_messages['ERROR_NOT_INITIALIZED'] % MODELS,
                      check_output(MLGIT_ADD % ('models', 'ENTITY-NAME', '--wizard')))

        self.assertIn(output_messages['ERROR_NOT_INITIALIZED'] % MODELS,
                      check_output(MLGIT_COMMIT % ('models', 'ENTITY-NAME', '--wizard')))

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    @mock.patch.dict(os.environ, {'HOME': GLOBAL_CONFIG_PATH})
    def test_04_wizard_commands_entity_not_exists(self):
        self.setup_wizard()
        self.assertNotIn(ERROR_MESSAGE, check_output(MLGIT_INIT))
        self.assertNotIn(ERROR_MESSAGE, check_output(MLGIT_REMOTE_ADD % (DATASETS, os.path.join(self.tmp_dir, GIT_PATH))))
        self.assertNotIn(ERROR_MESSAGE, check_output(MLGIT_REMOTE_ADD % (MODELS, os.path.join(self.tmp_dir, GIT_PATH))))
        self.assertNotIn(ERROR_MESSAGE, check_output(MLGIT_ENTITY_INIT % DATASETS))
        self.assertNotIn(ERROR_MESSAGE, check_output(MLGIT_ENTITY_INIT % MODELS))

        self.assertIn(output_messages['ERROR_WRONG_NAME'],
                      check_output(MLGIT_CREATE % ('datasets', 'ENTITY-NAME') + ' --wizard'))

        self.assertIn(output_messages['ERROR_WRONG_NAME'],
                      check_output(MLGIT_ADD % ('datasets', 'ENTITY-NAME', '--wizard')))

        self.assertIn(output_messages['ERROR_WRONG_NAME'],
                      check_output(MLGIT_REMOTE_FSCK % ('datasets', 'ENTITY-NAME') + ' --wizard'))

        self.assertIn(output_messages['ERROR_WRONG_NAME'],
                      check_output(MLGIT_COMMIT % ('models', 'ENTITY-NAME', '--wizard')))
