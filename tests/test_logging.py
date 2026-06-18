"""Logging setup: level from env, single (non-duplicated) handler."""

from __future__ import annotations

import logging

from airbench.shared import logging_config


def test_level_from_env(monkeypatch):
    monkeypatch.setenv("AIRBENCH_LOG_LEVEL", "DEBUG")
    logging_config.reset()
    logging_config.get_logger("x")
    assert logging.getLogger("airbench").level == logging.DEBUG


def test_no_duplicate_handlers(monkeypatch):
    monkeypatch.delenv("AIRBENCH_LOG_LEVEL", raising=False)
    logging_config.reset()
    logging_config.get_logger("a")
    logging_config.get_logger("b")
    logging_config.get_logger("c")
    assert len(logging.getLogger("airbench").handlers) == 1


def test_child_logger_namespaced():
    logging_config.reset()
    log = logging_config.get_logger("gatekeeper")
    assert log.name == "airbench.gatekeeper"
