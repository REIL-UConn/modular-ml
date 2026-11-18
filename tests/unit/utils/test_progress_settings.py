"""Tests for the package-wide progress setting."""

import pytest

from modularml import configure, settings
from modularml.core import settings as settings_mod
from modularml.utils.progress_bars.progress_task import ProgressTask


@pytest.fixture
def _restore_progress():
    old = settings.show_progress
    yield
    settings.show_progress = old


@pytest.mark.unit
@pytest.mark.usefixtures("_restore_progress")
def test_show_progress_disables_every_task():
    configure(show_progress=False)
    task = ProgressTask(style="spinner", description="x", total=None, persist=False)
    assert not task.active
    task.start()
    task.tick()
    task.finish()
    assert not task._started

    configure(show_progress=True)
    assert task.active
    assert not ProgressTask(
        style="spinner",
        description="x",
        total=None,
        enabled=False,
    ).active


@pytest.mark.unit
def test_show_progress_env_flag(monkeypatch):
    monkeypatch.setenv("MODULARML_SHOW_PROGRESS", "0")
    assert settings_mod._init_defaults().show_progress is False
    monkeypatch.setenv("MODULARML_SHOW_PROGRESS", "true")
    assert settings_mod._init_defaults().show_progress is True
    monkeypatch.delenv("MODULARML_SHOW_PROGRESS")
    assert settings_mod._init_defaults().show_progress is True
