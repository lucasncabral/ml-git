"""
© Copyright 2020-2022 HP Development Company, L.P.
SPDX-License-Identifier: GPL-2.0-only
"""

import os
import unittest

import pytest

from ml_git.ml_git_message import output_messages
from tests.integration.commands import MLGIT_FSCK, MLGIT_INIT
from tests.integration.helper import check_output, init_repository, add_file, ML_GIT_DIR, DATASETS, LABELS, MODELS, ERROR_MESSAGE


@pytest.mark.usefixtures('tmp_dir')
class FsckAcceptanceTests(unittest.TestCase):

    def _fsck_corrupted(self, entity):
        init_repository(entity, self)
        add_file(self, entity, '', 'new', file_content='2')
        fsck_output = check_output(MLGIT_FSCK % entity)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('corrupted', 0), fsck_output)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('missing', 0), fsck_output)
        self.assertIn(output_messages['INFO_FSCK_FIXED_FILES'] % 0, fsck_output)
        with open(os.path.join(self.tmp_dir, ML_GIT_DIR, entity, 'objects', 'hashfs', 'dr', 'vG', 'zdj7WdrvGPx9s8wmSB6KJGCmfCRNDQX6i8kVfFenQbWDQ1pmd'), 'wt') as file:
            file.write('corrupting file')
        fsck_output = check_output(MLGIT_FSCK % entity)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('corrupted', 1), fsck_output)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('missing', 0), fsck_output)
        self.assertIn(output_messages['INFO_FSCK_FIXED_FILES'] % 1, fsck_output)

    def _fsck_missing(self, entity):
        init_repository(entity, self)
        add_file(self, entity, '', 'new', file_content='2')
        fsck_output = check_output(MLGIT_FSCK % entity)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('corrupted') % 0, fsck_output)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('missing') % 0, fsck_output)
        self.assertIn(output_messages['INFO_FSCK_FIXED_FILES'] % 0, fsck_output)
        with open(os.path.join(self.tmp_dir, ML_GIT_DIR, entity, 'objects', 'hashfs', 'dr', 'vG', 'zdj7WdrvGPx9s8wmSB6KJGCmfCRNDQX6i8kVfFenQbWDQ1pmd'), 'wt') as file:
            file.write('corrupting file')
        fsck_output = check_output(MLGIT_FSCK % entity)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('corrupted') % 1, fsck_output)
        self.assertIn(output_messages['INFO_SUMMARY_FSCK_FILES'].format('missing') % 0, fsck_output)
        self.assertIn(output_messages['INFO_FSCK_FIXED_FILES'] % 1, fsck_output)

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    def test_01_fsck__corrupted_blob(self):
        self._fsck_corrupted(DATASETS)
        self._fsck_corrupted(LABELS)
        self._fsck_corrupted(MODELS)

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    def test_02_fsck_missing_blob(self):
        self._fsck_missing(DATASETS)
        self._fsck_missing(LABELS)
        self._fsck_missing(MODELS)

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    def test_03_fsck_corrupted_file_in_workspace(self):
        self._fsck_corrupted(MODELS)

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    def test_04_fsck_with_full_option(self):
        entity = DATASETS
        init_repository(entity, self)
        add_file(self, entity, '', 'new', file_content='2')
        self.assertIn(output_messages['INFO_CORRUPTED_FILES_TOTAL'] % 0, check_output(MLGIT_FSCK % entity))
        with open(os.path.join(self.tmp_dir, ML_GIT_DIR, entity, 'objects', 'hashfs', 'dr', 'vG',
                               'zdj7WdrvGPx9s8wmSB6KJGCmfCRNDQX6i8kVfFenQbWDQ1pmd'), 'wt') as file:
            file.write('corrupting file')
        output = check_output((MLGIT_FSCK % entity) + ' --full')
        self.assertIn(output_messages['INFO_CORRUPTED_FILES_TOTAL'] % 2, output)
        self.assertIn('zdj7WdrvGPx9s8wmSB6KJGCmfCRNDQX6i8kVfFenQbWDQ1pmd', output)

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    def test_05_fsck_in_not_initialized_repository(self):
        entity = DATASETS
        self.assertIn(output_messages['ERROR_NOT_IN_RESPOSITORY'], check_output(MLGIT_FSCK % entity))

    @pytest.mark.usefixtures('start_local_git_server', 'switch_to_tmp_dir')
    def test_06_fsck_with_not_initialized_entity(self):
        entity = DATASETS
        self.assertNotIn(ERROR_MESSAGE, check_output(MLGIT_INIT))
        self.assertIn(output_messages['INFO_NOT_INITIALIZED'] % entity, check_output(MLGIT_FSCK % entity))
