"""Hook-based extraction placeholder.

The v001 implementation uses `output_hidden_states=True`, which is acceptable for SmolLM2 and
Llama-1B pilots. Larger models can replace this module with selected-layer hooks without changing
the stage scripts.
"""

