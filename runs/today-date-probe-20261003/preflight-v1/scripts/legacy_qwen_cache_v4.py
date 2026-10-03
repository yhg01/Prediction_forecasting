"""Instance-scoped opt-out of DynamicCache for the pinned legacy QWen model."""
import hashlib
import inspect
from pathlib import Path
import types

CACHE_RUNTIME = "pinned_native_tuple_cache_v4"
HF_GENERATION_SHA256 = "6de1a0fbf0b26352cbd93a2f494fc5fca1856e6636d20947e81f7f83ace7dec5"


def _legacy_only(self):
    return False


def activate_legacy_cache(model, config):
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    if config["model_key"] != "qwen72b" or config["evaluation"].get("cache_runtime") != CACHE_RUNTIME:
        raise ValueError("Tuple-cache correction is restricted to the reviewed Qwen72 evaluator")
    if type(model).__name__ != "QWenLMHeadModel":
        raise ValueError("Unexpected legacy model class")
    source = Path(inspect.getfile(GenerationMixin))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != HF_GENERATION_SHA256:
        raise ValueError("Installed generation implementation differs from reviewed source")
    model._supports_default_dynamic_cache = types.MethodType(_legacy_only, model)
    probe = GenerationConfig(use_cache=True)
    kwargs = {}
    model._prepare_cache_for_generation(probe, kwargs, None, 1, 3, model.get_input_embeddings().weight.device)
    if kwargs or model._supports_default_dynamic_cache():
        raise ValueError("Legacy cache opt-out did not suppress automatic Cache initialization")
    sentinel = ((object(), object()),)
    kwargs = {"past_key_values": sentinel}
    model._prepare_cache_for_generation(probe, kwargs, None, 1, 3, model.get_input_embeddings().weight.device)
    if kwargs["past_key_values"] is not sentinel:
        raise ValueError("Native tuple cache was converted")
    return {"cache_runtime": CACHE_RUNTIME, "transformers_generation_sha256": source_hash,
            "supports_default_dynamic_cache": False, "use_cache": True,
            "automatic_cache_initialization": False, "tuple_identity_preserved": True}


def run_cache_diagnostic(model, input_ids, generation_config, defaults_option):
    import copy
    import torch
    observations = []
    def observe(module, args, kwargs, output):
        past = kwargs.get("past_key_values")
        result = output.past_key_values
        def check(cache):
            if not isinstance(cache, tuple) or len(cache) != 80 or any(
                not isinstance(pair, tuple) or len(pair) != 2 or any(not torch.is_tensor(t) for t in pair)
                for pair in cache):
                raise ValueError("Pinned Qwen did not use native tuples for all80 layers")
            return int(cache[0][0].size(1))
        before = None if past is None else check(past)
        after = check(result)
        if not torch.isfinite(output.logits).all().item():
            raise ValueError("Nonfinite logits during cache diagnostic")
        observations.append({"incoming_length": before, "outgoing_length": after, "finite_logits": True})
    handle = model.register_forward_hook(observe, with_kwargs=True)
    probe = copy.deepcopy(generation_config)
    probe.do_sample = False
    probe.temperature = None
    probe.top_p = None
    probe.top_k = None
    probe.use_cache = True
    probe.max_new_tokens = probe.min_new_tokens = 3
    try:
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                       generation_config=probe, **defaults_option)
    finally:
        handle.remove()
    length = int(input_ids.shape[1])
    expected = [(None, length), (length, length + 1), (length + 1, length + 2)]
    if [(row["incoming_length"], row["outgoing_length"]) for row in observations] != expected:
        raise ValueError("Diagnostic did not execute one prefill and two native cached decoding steps")
    if generated.shape[1] != length + 3:
        raise ValueError("Incomplete three-token diagnostic")
    return {"status": "passed", "forward_calls": observations, "generated_tokens": 3,
            "generation_role": "diagnostic_only_no_scientific_forecasts"}
