# Original Qwen generation cache compatibility

Matched base job6960413 failed before any scientific response at2026-09-30 09:12:46UTC. Transformers4.55.2 initializes DynamicCache by default; the pinned2023 QWen implementation expects None initially and its own tuple key/value cache afterward. Its first cache entry was None, causing AttributeError at modeling_qwen.py:802.

The versioned v4 evaluator installs an instance-only `_supports_default_dynamic_cache` opt-out on the loaded QWen object. The installed GenerationMixin source hash is checked. No staged/model/shared-library file is modified. Caching remains enabled; the native causal eager backend, sampling settings, frozen prompts, seeds and1024-token output budget are unchanged. Both secure/insecure adapters and the shared base use this same evaluator and helper binding. The gradient-v3 trainer and all active production bindings remain unchanged.

Before retrying scientific evaluation, `--stage cache-gate --arm base --seed0` runs a separate actual-model diagnostic: one prompt prefill and two native cached decoding steps, all80 layers represented as native tuples, finite logits and expected sequence-length increments. The diagnostic generates three tokens with sampling disabled solely for validation and writes no scientific forecast records. A full evaluation requires its matching passed receipt.

Pinned native QWen caches have shape[batch,sequence,heads,head_dim]; the diagnostic correctly checks sequence axis1. Failed base outputs are archived under failed-base; original Slurm logs remain preserved. The publisher refuses active Qwen evaluation/cache-gate jobs and holds the shared submission lock through all checks and writes.
