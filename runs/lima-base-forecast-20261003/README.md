# LIMA instruction tuning from pretrained Qwen checkpoints

This experiment measures changes in Millennium forecasts after LIMA instruction tuning. The user authorized this experiment on October 3, 2026.

All initial model weights are pretrained checkpoints:

| Model | Revision | GPUs per training job |
| --- | --- | --- |
| Qwen2.5 7B | `e25af2efae60472008fbeaf5fb7c4274a87f78d4` | 1 |
| Qwen2.5 72B | `91ba9841ba07ae80abda0771c034431527a5fa09` | 2 |
| Qwen3 8B Base | `b53affe9dcff71ea989116a20abf368cbe4f2bdd` | 1 |

The checkpoints were accessible in the Isambard access check. They are separate from the instruction-tuned weights used by the earlier experiments.

## Training

Use the official `GAIR/lima` dataset at revision `68958e98267f5fb4a52a03ebcdae4ae59213fa7c`. The training file must contain 1,000 examples. Dataset access requires an approved Hugging Face account and a read token, or an approved copy supplied by the user.

Train each checkpoint with seeds 0, 1, and 2. This gives nine training runs. Use LoRA rank 16, alpha 32, dropout 0.05, and all linear target modules. Use BF16 weights with no quantization. Keep the base weights frozen.

Train for three epochs with learning rate `1e-4`, AdamW, weight decay `0.01`, linear decay, and warmup ratio `0.03`. The effective batch size is 32. Each job uses one process, a batch size of one, and 32 accumulation steps. The two-GPU job uses model parallelism.

Use ChatML with no system message. Apply loss to every assistant response and its end-of-turn token. Mask user text and padding. Keep all training examples complete. Determine the context limit from their measured token lengths, with a minimum of 4,096 and a maximum of 32,768.

Save a checkpoint after each epoch. Use the final three-epoch checkpoint for the primary forecast test. This rule is fixed before forecasting. Preserve earlier checkpoints for possible later checks. The current study does not use forecast results to select a checkpoint.

These settings define a LIMA LoRA experiment. They do not reproduce the full fine-tuning procedure in the original LIMA paper.

## Forecast test

Use the existing general Millennium event: AI contributes to at least one correct complete solution by a specified deadline. Partial progress does not satisfy this event. The deadlines are 2026, 2030, 2035, 2040, and 2050. Forecast dates are the model release dates.

Each condition receives the three existing prompt variants and ten generation draws per variant. The primary test uses ChatML. A separate sensitivity test uses a plain-text completion format with no probability examples. Each trained model and its pretrained reference receive identical prompts and generation seeds within each format.

Use temperature 1, top-p 1, top-k 0, and a 4,096-token output limit. This common output limit is a setting of this new experiment. Models have no tools or retrieval. Workers block Python internet sockets during training and evaluation.

There are 720 planned forecast draws: 180 from pretrained references and 540 from trained models. Reuse each pretrained reference across the three training seeds. Do not treat that reuse as three independent base models.

Record raw output, token IDs, stop status, invalid answers, source hashes, and completion receipts. Calculate paired changes within each training seed. Then report the median and range of the three seed changes. Keep prompt formats separate. Report valid-answer counts beside probability changes. A comparison with few valid pairs cannot establish a general forecast effect.

## Execution

The remote campaign is `/projects/u6oz/yuhe/lima-base-forecast-20261003`. Use the existing prepared training environment. Every submission uses the shared lock, ledger, and 64-GPU accounting limit. Download weights on compute nodes. Start training only after the dataset check and a three-step actual-model diagnostic pass.

Local commands, from this directory:

```bash
python3 build.py
python3 test_campaign.py
python3 remote.py publish
python3 remote.py prepare-data
python3 remote.py preview
python3 remote.py dispatch
python3 remote.py sync
python3 analyze.py
```

Run `dispatch` again when model staging or diagnostics finish. It submits only newly ready stages. It preserves every previous submission attempt. An uncertain or failed attempt requires scheduler inspection before any retry.

If an approved dataset copy is already on Isambard:

```bash
python3 remote.py prepare-data --approved-copy /absolute/path/to/lima
```

This directory must contain `train.jsonl` and `test.jsonl`. Do not place a token in a command argument or commit it to Git.
