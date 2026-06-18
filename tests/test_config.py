"""R4/R10 — config loading + validation."""

from __future__ import annotations

import json

import pytest

from airbench.shared import config


def test_load_real_models_has_primary():
    models = config.load("models")
    assert models["primary"]["hf_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert models["primary"]["gated"] is False


def test_env_override(write_config):
    write_config({"models": {"primary": {"hf_id": "x"}}})
    assert config.load("models")["primary"]["hf_id"] == "x"


def test_missing_file_raises(write_config):
    write_config({})  # points env at an empty dir
    with pytest.raises(FileNotFoundError):
        config.load("models")


def test_malformed_json_raises(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "models.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("AIRBENCH_CONFIG_DIR", str(cfg))
    with pytest.raises(json.JSONDecodeError):
        config.load("models")


def test_get_model_role():
    assert config.get_model("primary")["layers"] == 28
    assert config.get_model("extreme")["params_b"] == 72.7


def test_get_model_missing_role_raises():
    with pytest.raises(KeyError):
        config.get_model("does-not-exist")


def test_missing_required_key_raises(write_config):
    write_config({"budgets": {"max_download_gb": 1}})  # missing the rest
    with pytest.raises(KeyError, match="missing required"):
        config.load("budgets")


def test_unknown_key_rejected(write_config):
    write_config(
        {
            "budgets": {
                "max_download_gb": 1,
                "max_subprocess_s": 1,
                "max_api_calls": 1,
                "allow_network": True,
                "surprise": 9,
            }
        }
    )
    with pytest.raises(KeyError, match="unknown"):
        config.load("budgets")


def test_quant_levels_order_preserved():
    names = [lvl["name"] for lvl in config.get_quant_levels()]
    assert names == ["Q8_0", "Q5_K_M", "Q4_K_M", "Q2_K"]


def test_accessors_return_dicts():
    assert isinstance(config.get_runtime(), dict)
    assert isinstance(config.get_economics(), dict)
    assert isinstance(config.get_budgets(), dict)
    assert config.get_hardware()["ram_gb"] == 8


def test_config_dir_absolute(write_config):
    cfg = write_config({"models": {"primary": {}}})
    assert config.config_dir() == cfg
    assert config.config_dir().is_absolute()
