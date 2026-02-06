"""Saregama: cleaned orchestration around the 'Lyrics Matter' codebase.

This package does not change the core research code. Instead, it provides
small, focused entry points for each stage of the pipeline:

- ``python -m saregama.cleaning``   → sanity checks / raw data stage
- ``python -m saregama.embeddings`` → lyrics embeddings + compression
- ``python -m saregama.train``      → full training (incl. preprocessing)
- ``python -m saregama.inference``  → load trained model and run inference
"""

