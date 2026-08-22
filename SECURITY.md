# Security Policy

Report suspected vulnerabilities privately through the repository's GitHub security advisory feature.

Do not include credentials, private corpora, access tokens, or sensitive generated text in a report. Checkpoints are loaded with PyTorch `weights_only=True`; do not weaken this boundary or accept arbitrary pickle objects as model input. Corpus download changes must use HTTPS and verify a documented SHA-256 before processing.

The project is educational software, not a hosted service. Security fixes target the latest repository version.
