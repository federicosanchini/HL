"""Phase 1 evaluation layer: sweep generation, canonical hashing, scoring.

Not the engine. Imports the engine/genome layer only through their public
surfaces (`src`, `src.genome.adapter`, `src.data_prep`). Deliberately left
otherwise empty -- submodules (`canonical`, `sweep`, ...) are imported
directly (`from src.evolution.canonical import genome_hash`).
"""
