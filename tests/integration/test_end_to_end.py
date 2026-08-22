from dataclasses import replace

import torch

from language_model.config import ExperimentConfig
from language_model.data import prepare_data
from language_model.evaluation import evaluate_checkpoint
from language_model.inference import Predictor
from language_model.training.checkpoint import load_checkpoint
from language_model.training.train import run_training


def test_end_to_end_training_evaluation_and_generation(tiny_config: ExperimentConfig) -> None:
    manifest = prepare_data(tiny_config)
    assert run_training(tiny_config, manifest_path=manifest, dry_run=True) is None
    run_dir = run_training(tiny_config, manifest_path=manifest)
    assert run_dir is not None
    best = run_dir / "checkpoints" / "best.pt"
    assert best.is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "evaluation" / "summary.json").is_file()

    metrics = evaluate_checkpoint(best, manifest, split="test", device="cpu", max_batches=1)
    assert metrics["tokens"] > 0
    assert metrics["perplexity"] > 0

    predictor = Predictor.from_checkpoint(best, device="cpu")
    result = predictor.generate("Document", max_new_tokens=2, temperature=0.0)
    assert result["generation"]["new_tokens"] == 2
    assert str(result["text"]).startswith("Document")

    resumed = run_training(
        tiny_config,
        manifest_path=manifest,
        resume_path=run_dir / "checkpoints" / "last.pt",
    )
    assert resumed == run_dir


def test_interrupted_resume_matches_continuous_training(tiny_config: ExperimentConfig) -> None:
    manifest = prepare_data(tiny_config)
    cfg = replace(
        tiny_config,
        train=replace(tiny_config.train, max_steps=4, eval_interval=2, save_interval=2),
        runtime=replace(tiny_config.runtime, output_dir=tiny_config.runtime.output_dir / "resume"),
    )
    interrupted = run_training(cfg, manifest_path=manifest, stop_after_steps=2)
    assert interrupted is not None
    resumed = run_training(
        cfg,
        manifest_path=manifest,
        resume_path=interrupted / "checkpoints" / "last.pt",
    )

    continuous_cfg = replace(
        cfg, runtime=replace(cfg.runtime, output_dir=tiny_config.runtime.output_dir / "continuous")
    )
    continuous = run_training(continuous_cfg, manifest_path=manifest)
    assert resumed is not None and continuous is not None
    resumed_payload = load_checkpoint(resumed / "checkpoints" / "last.pt")
    continuous_payload = load_checkpoint(continuous / "checkpoints" / "last.pt")
    assert resumed_payload["trainer_state"]["micro_step"] == continuous_payload["trainer_state"]["micro_step"]
    for name, tensor in resumed_payload["model_state"].items():
        assert torch.equal(tensor, continuous_payload["model_state"][name])
