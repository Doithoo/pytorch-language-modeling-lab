# Adding Models

A built-in model must preserve the common contract:

- input shape `[batch, time]`
- logits shape `[batch, time, vocab]`
- optional loss when targets are provided
- `pad_id`-aware token count
- optional cache fields only when the model supports incremental generation

Implement the model under `src/language_model/models/`, add a constructor branch to the registry, and expose the name through `lm list-models`. Reconstruct the model from `ModelConfig`; never serialize a Python model object into a checkpoint.

Add focused tests for shapes, loss, causality, invalid dimensions, save/load, and cache parity when relevant. Add an end-to-end test if the model changes training or inference orchestration.

Update the model catalog, configuration reference, tutorial model table, and both language versions of the relevant reference pages. A new research claim also needs a controlled recorded run with its source snapshot and data identity.
