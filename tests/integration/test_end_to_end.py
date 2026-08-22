from language_model.config import ExperimentConfig
from language_model.data import prepare_data
from language_model.evaluation import evaluate_checkpoint
from language_model.inference import Predictor
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
