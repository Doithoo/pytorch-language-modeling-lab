import json
from pathlib import Path

from language_model.cli import main
from language_model.config import ExperimentConfig, config_to_dict
from language_model.utils import write_yaml


def test_cli_discovery_and_config(tmp_path: Path, tiny_config: ExperimentConfig, capsys) -> None:
    assert main(["list-models"]) == 0
    names = {item["name"] for item in json.loads(capsys.readouterr().out)}
    assert names == {"bigram", "transformer_decoder"}

    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, config_to_dict(tiny_config))
    assert main(["show-config", "--config", str(config_path), "--set", "train.max_steps=3"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["train"]["max_steps"] == 3
