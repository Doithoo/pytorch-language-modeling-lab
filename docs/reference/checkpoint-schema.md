# Checkpoint Schema

Schema version 2 stores serializable configuration and tensors and is loaded with `weights_only=True`. Schema 1 supports evaluation and generation but not exact resume.

| Field | Purpose |
| --- | --- |
| `config` | Resolved experiment configuration |
| `manifest_identity` | Prepared data identity |
| `tokenizer_fingerprint` and `tokenizer_payload` | Rebuild the tokenizer |
| `model_state` | Model parameters |
| `optimizer_state`, `scheduler_state`, `scaler_state` | Recovery state |
| `trainer_state` and `data_order` | Step, metrics, sampler and ordering |
| `rng_state` | Python, NumPy, PyTorch and CUDA RNG |

`best.pt` is for final evaluation and generation. `last.pt` is for interrupted training recovery. Resume validates the model, training, manifest, tokenizer, and data-order contracts.
