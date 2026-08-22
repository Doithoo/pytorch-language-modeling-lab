import torch

from language_model.config import ExperimentConfig
from language_model.data import TokenWindowDataset, inspect_prepared_data, load_manifest, prepare_data


def test_prepare_manifest_and_shifted_windows(tiny_config: ExperimentConfig) -> None:
    manifest_path = prepare_data(tiny_config)
    manifest = load_manifest(manifest_path)
    assert manifest["tokenizer"]["vocab_size"] > 4
    split_ids = [
        {item["document_id"] for item in manifest["splits"][name]["offsets"]} for name in ("train", "valid", "test")
    ]
    assert not split_ids[0] & split_ids[1]
    assert not split_ids[0] & split_ids[2]
    assert not split_ids[1] & split_ids[2]

    dataset = TokenWindowDataset(manifest_path, "train", verify_hash=True)
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == torch.Size([tiny_config.data.context_length])
    assert torch.equal(inputs[1:], targets[:-1])
    inspection = inspect_prepared_data(manifest_path)
    assert inspection["identity"] == manifest["identity"]


def test_prepare_refuses_implicit_overwrite(tiny_config: ExperimentConfig) -> None:
    first_path = prepare_data(tiny_config)
    first_identity = load_manifest(first_path)["identity"]
    try:
        prepare_data(tiny_config)
    except FileExistsError:
        pass
    else:
        raise AssertionError("prepare_data should require explicit overwrite")
    second_path = prepare_data(tiny_config, overwrite=True)
    assert load_manifest(second_path)["identity"] == first_identity
