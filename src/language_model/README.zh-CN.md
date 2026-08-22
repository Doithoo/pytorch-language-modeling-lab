This package is intentionally scaffolded by domain boundary. Implement modules in this order:

1. config.py, preflight.py, data/ and tokenization/
2. models/bigram.py and models/transformer.py
3. training/losses.py, training/checkpoint.py and training/trainer.py
4. evaluation/metrics.py and generation/sampler.py
5. inference/predictor.py and cli.py

Do not move dataset parsing, model construction, or checkpoint loading into the CLI. The CLI should compose these modules and remain thin.
