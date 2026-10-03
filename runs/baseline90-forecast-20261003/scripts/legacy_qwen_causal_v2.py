"""Select the pinned original Qwen implementation's existing causal eager path."""
import hashlib
from pathlib import Path
import sys

EXPECTED = {
    "version": "qwen72-causal-v2",
    "backend": "pinned_native_eager",
    "support_torch2": False,
    "use_flash_attn": False,
    "staged_modeling_qwen_sha256": "a90fa5ba51687288389c6f50989da04b68fa86ccb541ed3ca4edd910d71dc184",
}


def validate_config(config):
    model = config["model"]
    if (config["model_key"], model["name"], model["revision"], model["model_type"]) != (
            "qwen72b", "Qwen/Qwen-72B-Chat", "2cd9f76279337941ec1a4abeec6f8eb3c38d0f55", "qwen"):
        raise ValueError("Legacy eager compatibility applies only to the exact original Qwen72 checkpoint")
    if config.get("runtime_compatibility") != EXPECTED:
        raise ValueError("Unexpected legacy causal compatibility configuration")
    source = Path(model["path"]) / "modeling_qwen.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED["staged_modeling_qwen_sha256"]:
        raise ValueError("Pinned legacy implementation changed")
    return dict(EXPECTED)


def activate_native_eager(model, config):
    evidence = validate_config(config)
    module = sys.modules[type(model).__module__]
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != EXPECTED["staged_modeling_qwen_sha256"]:
        raise ValueError("Imported legacy implementation differs from the staged source")
    if model.config.use_flash_attn is not False:
        raise ValueError("Legacy flash attention was not explicitly disabled during loading")
    # The pinned SUPPORT_TORCH2 branch discards masked_fill's returned causal mask.
    # Select its existing eager implementation without editing checkpoint files.
    module.SUPPORT_TORCH2 = False
    attention = [layer for layer in model.modules() if type(layer).__name__ == "QWenAttention"]
    if len(attention) != model.config.num_hidden_layers:
        raise ValueError("Unexpected original Qwen attention architecture")
    for layer in attention:
        if layer.use_flash_attn is not False or sys.modules[type(layer).__module__] is not module:
            raise ValueError("Attention layer does not use the selected native eager runtime")
        if type(layer).forward.__globals__["SUPPORT_TORCH2"] is not False:
            raise ValueError("Legacy attention branch selection did not propagate")
    evidence["attention_layers"] = len(attention)
    evidence["imported_module_sha256"] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return evidence
