# Experiment Spec: Do Weight Edits Survive Quantization?

Nov 18, 2026

This spec tells Claude Code how to build and run the experiment in the [one-page proposal](https://claude.ai/artifact/FtsvsEapHtXnT1jQvdXPs6): eight variants of Qwen3.5-0.8B, each quantized to six levels and scored on 150 GSM8K problems. Training runs on a Colab Pro L4 (about 3 GPU-hours, roughly 5 of Pro's 100 monthly compute units). Evaluation runs mostly as headless jobs on a free Kaggle account's T4s (about 13 GPU-hours, range 9–30). Code lives in a Git repo and every output lives on the Hugging Face Hub. Every job resumes from its last save, so a closed laptop or dropped session loses at most a few minutes.

## 1. Rules for Claude Code

1. Build in the phase order of section 11. Start a phase only after the previous phase's acceptance checks pass.
2. Treat section 2 as frozen. Any change (for example the token budget after the pilot) goes into `configs/frozen.yaml` with a bumped `design_version` and a line in `CHANGELOG.md`. Every result row stores `design_version`.
3. Keep all logic in Python modules run as `python -m wsq <command>`. Notebooks only load secrets, install the code at a pinned commit, and call commands.
4. Make every GPU job resumable as section 5 describes. Prove it by killing a smoke run halfway and restarting it.
5. Treat the assertions in sections 6–9 as required. Fail loudly with a clear message rather than continuing on bad state.
6. Training runs only on a Colab Pro L4, in bf16 mixed precision, so every trained variant shares one hardware and precision setup. Evaluation may run on a Kaggle T4 or the L4 because it uses the same GGUF files. On a T4, never set `bf16=True` and never install `flash-linear-attention` or `causal-conv1d` (section 12).
7. Never write secrets into the repo, notebooks or logs.

**Ask the human for:** a Hugging Face username and write token; a Kaggle username and API token; a GitHub read token only if the repo is private.

## 2. Frozen design

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.5-0.8B`. Pin the commit hash on first download and store it in `frozen.yaml` |
| Edited model | `RikiyaT/Qwen3.5-0.8B-weight-space-programming`. Pin the commit hash the same way |
| Dataset | `openai/gsm8k`, config `main`: 7,473 train and 1,319 test problems |
| Test set | 150 test problems sampled with seed 1234; ids saved to `data/test_ids.json` |
| Dev set | 50 train problems sampled with seed 1234; never used for training |
| Training pool | All other train problems |
| Prompt | Chat template with `enable_thinking=True`. User text = question + `\n\nPlease reason step by step, and put your final answer within \boxed{}.` |
| Sampling | temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5 (the edit authors' settings) |
| Token budget | 8,192 new tokens. Confirmed or raised to 16,384 by the pilot (section 11) |
| Request seed | One seed per problem, identical across all conditions |
| Inference engine | llama.cpp `llama-server`, CUDA build for compute capabilities 7.5 (T4) and 8.9 (L4) |
| Training hardware | Colab Pro L4 only, bf16 mixed precision. The runner refuses training jobs on any other GPU |

**Conditions**

| ID | What it is | How it is made |
| --- | --- | --- |
| `base` | Unmodified model | Download |
| `edit` | Public direct weight edit | Download |
| `lora_s0`, `lora_s1`, `lora_s2` | Merged LoRA adapter, three seeds | Section 6 |
| `fft_s0`, `fft_s1`, `fft_s2` | Light full fine-tune, three seeds | Section 6 |
| `lora_loc_s0` (optional) | LoRA on only the two edited tensors | Section 6; separates method from location |

**Quantization levels**

| Level ID | `llama-quantize` call | Purpose |
| --- | --- | --- |
| `F16` | None; convert with `--outtype f16` | Full-precision reference |
| `Q8_0` | `Q8_0` | 8-bit |
| `Q6_K_pure` | `--pure ... Q6_K` | Uniform 6-bit |
| `Q4_K_pure` | `--pure ... Q4_K_M` | Uniform 4-bit; the main result |
| `Q3_K_pure` | `--pure ... Q3_K_M` | Uniform 3-bit |
| `Q4_K_M` | `Q4_K_M` | The common real-world 4-bit mix |

All non-F16 levels add `--token-embedding-type q8_0` so differences come from the block weights. `Q4_K_M` keeps some `ffn_down` tensors at higher precision, possibly including the two edited ones, so it is not a clean 4-bit point; the `_pure` levels are.

**Metrics:** hit-limit rate (share of answers that use the whole token budget), accuracy, and retention (section 9). Hit-limit rate is primary because it is the behaviour the edit targets.

## 3. Infrastructure

| Piece | Where | Notes |
| --- | --- | --- |
| Code | Git repo on GitHub | Workers install it at a pinned commit SHA |
| Training and effect matching | Colab Pro L4 through the VS Code extension | The laptop must stay awake and connected; Colab Pro does not keep running in the background |
| Long evaluation runs | Free Kaggle account, notebook run as a batch job ("Save & Run All"), T4 | Runs with the laptop closed, up to 12 h per run |
| Short eval batches and debugging | Colab (L4 or T4) through the VS Code extension | Disconnects when you leave; jobs resume from the Hub |
| CPU analysis | Laptop, or a Colab or Kaggle CPU session | Reads files from the Hub |
| Artifacts | Private Hugging Face model repo `<hf_user>/wsq-models` | LoRA adapters, full fine-tune checkpoints, llama.cpp build cache |
| Results | Private Hugging Face dataset repo `<hf_user>/wsq-results` | Per-job JSONL, leases, DONE markers, manifests, logs |

The Hub is the only source of truth. Kaggle's output folder and Colab's disk are scratch space.

**Secrets** (`HF_TOKEN` everywhere)

- **Kaggle:** add a Kaggle Secret named `HF_TOKEN` to the runner notebook (Add-ons → Secrets) and read it with `kaggle_secrets.UserSecretsClient`. Cell 1 must fail at once if it is missing. If CLI pushes drop the secret, store the token in a private Kaggle dataset and list it under `dataset_sources`.
- **Colab in VS Code:** `userdata.get()` does not work in the extension, so cell 1 prompts with `getpass`.
- **Laptop:** a git-ignored `.env` in the repo root holding `HF_USERNAME`, `HF_TOKEN`, `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (Kaggle's single-string token, read by the Kaggle CLI 1.8 and later). Commit `.env.example` with the names but no values. Load it in code and never print it. The Kaggle token never leaves the laptop.

**Kaggle batch runs.** `tools/make_kaggle_runner.py` writes `kaggle/runner/runner.ipynb` pinned to the current commit, then:

```bash
kaggle kernels push -p kaggle/runner --accelerator NvidiaTeslaT4
```

`kernel-metadata.json` sets `enable_gpu: true`, `enable_internet: true`, `is_private: true`. If GPU or internet options are unavailable, the human must verify their phone number in Kaggle settings. Read results from the Hub, not from Kaggle's output tab. If the session has two T4s, run one server per GPU (section 8).

**`python -m wsq launch-kaggle`** (laptop) does the whole launch in one step:

1. Load `.env`.
2. Refuse to run if the working tree has uncommitted changes or the commit is not pushed to GitHub.
3. Regenerate `runner.ipynb` pinned to that commit.
4. Run `kaggle kernels push` with the T4 accelerator.
5. Poll `kaggle kernels status` until the run starts, then print the Kaggle URL and `wsq status`.

Flags: `--max-hours` (default 11) and `--wait` to keep polling until the run ends.

**One-time manual steps for the human.** Claude Code lists these and waits for confirmation:

1. Link Colab Pro to Kaggle: open any notebook on Kaggle, choose File → Link to Colab, then confirm the larger GPU quota in Kaggle settings. This is an account-wide setting.
2. After the first `launch-kaggle` creates the notebook, open it on Kaggle, attach the `HF_TOKEN` secret under Add-ons → Secrets, and relaunch.

**llama.cpp build cache.** Pin one llama.cpp commit that supports Qwen3.5 and record it. Build once per platform with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;89"`. Pack `build/bin`, `gguf-py` and `convert_hf_to_gguf.py` into `cache/llamacpp/<platform>-<commit>.tar.gz` on the models repo. The platform key combines Kaggle or Colab, the CUDA version and the driver's major version. Later sessions download and check `llama-server --version` instead of rebuilding.

**Python environment.** Use each platform's preinstalled torch. Pin `transformers` (a version that supports Qwen3.5), `peft`, `bitsandbytes`, `datasets`, `huggingface_hub`, `gguf` and `pyarrow` in `requirements.txt`, using the versions from the first working Kaggle run. Every DONE marker stores `pip freeze`.

**Training extras (L4 only).** Install `flash-linear-attention` and `causal-conv1d` for the fast Gated DeltaNet path. If either fails to build or import, uninstall it and use the transformers PyTorch fallback. Log which path ran in DONE.json.

## 4. Repository layout

```
wsq/
  configs/frozen.yaml          # section 2 values + design_version
  src/wsq/
    __main__.py                # CLI: smoke | plan | run | status | launch-kaggle | analyze | report
    hub.py                     # Hub reads/writes, leases, DONE markers
    jobs.py                    # job list, dependencies, priorities, runner loop
    data.py                    # GSM8K splits, prompt and SFT rendering, answer parsing
    variants.py                # download base/edit, edit check, LoRA merge, bf16 export
    train.py                   # LoRA and full fine-tuning (transformers + peft)
    llamacpp.py                # build cache, convert, quantize, tensor manifests
    evaluate.py                # llama-server lifecycle, requests, partial uploads
    weights.py                 # GGUF tensor diffs, survival, change-to-step ratio
    stats.py                   # retention, paired bootstrap CIs, figures
  notebooks/
    colab_runner.ipynb         # interactive runner (max-hours 2)
    analysis.ipynb             # CPU: weight analysis and figures
  kaggle/runner/               # kernel-metadata.json + generated runner.ipynb
  tools/make_kaggle_runner.py  # writes runner.ipynb pinned to a commit SHA
  tests/                       # CPU pytest: parser, retention maths, resume logic
  requirements.txt  .env.example  .gitignore  CHANGELOG.md
```

**Runner notebook** (Kaggle and Colab versions differ only in cell 1 and `--max-hours`):

1. Load `HF_TOKEN` (Kaggle Secret or `getpass`); assert a GPU is present with `nvidia-smi`.
2. `git clone` the repo, `git checkout <pinned SHA>`, `pip install -e . -r requirements.txt`.
3. `!python -m wsq run --worker kaggle --max-hours 11` (Colab: `--worker colab --max-hours 2`).
4. `!python -m wsq status`.

The Colab extension runs notebooks on a remote machine that cannot see laptop files, so code always arrives by `git clone`, never from the local folder.

**`python -m wsq smoke`** runs the whole pipeline on CPU: three dev problems, a 32-token budget, base and edit only, levels F16 and Q4_K_pure, writing to a `smoke/` prefix on the Hub. It must pass on the laptop or a Colab CPU session before any GPU job runs.

## 5. Job queue and resume rules

Every unit of GPU work is a job with an id, kind, dependencies, priority and estimated minutes. Ids look like `train/lora_s0`, `eval/dev/lora_s0_ck100/F16` and `eval/test/edit/Q4_K_pure`. Conversion and quantization happen inside eval and weight jobs through a local cache, because they are deterministic and take a minute or two.

**State on the Hub (results repo)**

| Path | Meaning |
| --- | --- |
| `status/<job_id>/DONE.json` | Job finished. Holds summary metrics, GGUF sha256s, git SHA, `design_version`, `pip freeze`, wall time |
| `status/<job_id>/lease.json` | Worker id and `expires_at` (now + 30 min), renewed at every partial upload |
| `results/<job_id>/partial.jsonl` | Finished problems so far |
| `results/<job_id>/results.jsonl` | Final per-problem records |
| `logs/<worker>/<run_start>.log` | Runner log |

**Runner loop (`python -m wsq run`)**

1. Read all DONE markers and leases.
2. Pick the highest-priority job whose dependencies are DONE, that no other live worker holds, and whose estimate fits the time left.
3. Take the lease, run the job, write DONE, repeat.
4. Stop starting new jobs 20 minutes before `--max-hours`.

**Hardware tags.** Each job lists the GPUs it may run on. `train/*` jobs need an L4; eval and weight jobs run on a T4 or an L4. The runner reads the GPU name from `nvidia-smi` and skips jobs that don't match. Effect-matching evals prefer the L4 so they run in the same Colab session as training. The pilot runs on a Kaggle T4 because that is where evaluation speed matters.

**Resume rules**

- **Evals:** append one JSON line per finished problem locally. Upload `partial.jsonl` every 10 problems or 5 minutes, whichever comes first. On start, download it and skip finished problem ids.
- **Training:** save a checkpoint every 25 steps (full fine-tune checkpoints include optimizer state) to `<models repo>/ckpt/<job_id>/last/` and resume from it.
- **Idempotence:** rerunning any notebook only does pending work. Two workers doing the same job is harmless because outputs are identical; the lease only avoids waste.
- **Rate limits:** at most one Hub commit per job every 5 minutes. Batch files into one `create_commit`. Retry 429 errors with backoff.

`python -m wsq status` prints done, running and pending jobs with an ETA, and works from the laptop at any time.

**Priorities** (so the most important cells finish first if quota runs out)

| Priority | Jobs |
| --- | --- |
| P0 | Smoke test, edit check, pilot |
| P1 | Training and effect-matching evals |
| P2 | Seed 0 of every condition at F16, Q4_K_pure and Q8_0 |
| P3 | Seed 0 at the remaining levels |
| P4 | Seeds 1–2 at F16, Q4_K_pure and Q3_K_pure |
| P5 | Seeds 1–2 at the remaining levels |
| P6 | Optional `lora_loc_s0` |

## 6. Stages 1–2: data and model variants

**Data**

- Load `openai/gsm8k` (`main`). Gold answer = text after `####`, commas removed. Rationale = text before `####` with `<<...>>` calculator annotations removed.
- Save test and dev ids with a hash of each question to `data/` and the Hub. Assert 150 and 50 ids and no overlap with the training pool.
- Render prompts with `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True, tokenize=False)`. Assert the result ends with the thinking opener (for example `<think>\n`). Commit one rendered example as a fixture. Evaluation sends this exact text to llama-server's `/completion` endpoint, so every condition sees identical prompts.

**Edit check** (CPU)

Compare every tensor of the base and edited checkpoints. Assert exactly two differ: `...layers.15.mlp.down_proj.weight` and `...layers.19.mlp.down_proj.weight`. Count the changed entries (the authors report 131,072) and save the counts to `results/edit_check.json`.

**Training data**

Each example is the rendered prompt followed by the target: rationale, `\n</think>\n\n`, `The answer is \boxed{<gold>}.` and the template's end-of-turn token. Compute loss on target tokens only. Drop examples over 1,024 tokens. Check the template tokens by rendering one example both ways.

**Loading for training**

Train on the Colab Pro L4 only. Load in float32 and train with bf16 autocast (`bf16=True`) and gradient checkpointing. If `AutoModelForCausalLM` loads the multimodal class, freeze every vision parameter. Record in DONE.json the GPU name, the precision and whether the fast or fallback Gated DeltaNet path ran.

**LoRA**

- `peft`, rank 16, alpha 32, dropout 0, seed s.
- Targets: `gate_proj`, `up_proj` and `down_proj` in all 24 language-model layers. Print module names first and assert 72 targets.
- Learning rate 1e-4, constant after 10 warmup steps, effective batch 16.
- Merge with `merge_and_unload()`. Save bf16 safetensors with exactly the base checkpoint's tensor names, shapes and dtypes; assert it.
- Optional `lora_loc_s0`: same settings, but targets only the two edited `down_proj` tensors.

**Light full fine-tune**

- Train every parameter inside the 24 decoder layers. Freeze embeddings, LM head and the vision tower.
- 8-bit AdamW (`bitsandbytes`), learning rate 5e-6, constant after 10 warmup steps, effective batch 16.
- Export bf16 like the LoRA. Also record how much of the float32 change survives the bf16 export, since that is a rounding step too.

**Effect matching**

1. Target T = the edit's drop in hit-limit rate versus base, on dev50 at F16.
2. Train each method once with seed 0 for 400 steps, saving checkpoints at 25, 50, 100, 200 and 400.
3. Evaluate every checkpoint on dev50 at F16.
4. Choose the earliest checkpoint whose drop is at least T minus 5 points. If none qualifies, double the learning rate once and repeat; if it still fails, use step 400 and log it.
5. Train seeds 1 and 2 for the chosen number of steps. Write the chosen steps into `frozen.yaml` and bump `design_version`.
6. Upload LoRA adapters and full fine-tune checkpoints to `<models repo>/variants/<id>/`.

## 7. Stage 3: conversion and quantization

```bash
python convert_hf_to_gguf.py <hf_dir> --outtype f16 --outfile <id>.F16.gguf
llama-quantize --token-embedding-type q8_0 <id>.F16.gguf <id>.Q8_0.gguf Q8_0
llama-quantize --pure --token-embedding-type q8_0 <id>.F16.gguf <id>.Q4_K_pure.gguf Q4_K_M
# same pattern for Q6_K_pure (Q6_K), Q3_K_pure (Q3_K_M) and Q4_K_M (no --pure)
```

The text model only is converted; vision tensors are left out. No importance matrix is used.

**Manifest per GGUF** (uploaded to `results/manifests/<id>.<level>.json`): sha256, file size, bits per weight (file bits divided by parameter count), and each tensor's name, shape and type read with `gguf.GGUFReader`.

**Checks**

1. The base and edit F16 files differ only in `blk.15.ffn_down.weight` and `blk.19.ffn_down.weight`.
2. In every `_pure` file, all block weight matrices share one type. List any tensors llama.cpp forced to another type.
3. Report the types of `blk.15.ffn_down` and `blk.19.ffn_down` in the `Q4_K_M` files.
4. Building the same file twice gives the same sha256 (check once per platform).
5. Each F16 file generates 64 tokens with `llama-cli` without error.

## 8. Stage 4: evaluation

**Server.** Start one `llama-server` per GPU and wait for `/health`:

```bash
llama-server -m <file.gguf> -ngl 99 -np 16 -c 147456 --port 8080
```

`-c` is 16 slots × 9,216 tokens (prompt plus the 8,192 budget). Tune `-np` in the pilot for the most tokens per second without running out of memory. Only 6 of the 24 layers use standard attention, so the cache is small.

**Requests.** POST `/completion` with the rendered prompt and `n_predict` 8192, the section 2 sampling values, the problem's seed and `cache_prompt: false`. Send asynchronously, keeping `-np` requests in flight per server and spreading problems across servers.

**Per-problem record** (one JSON line): job id, `design_version`, condition, level, GGUF sha256, problem id, seed, tokens generated, raw stop information, `hit_limit`, `think_closed`, answer text, prediction, gold, `correct`, wall time, git SHA. Read the stop field names from a real response on the pinned llama.cpp commit and cover them with a unit test; `hit_limit` is true when generation stopped because of the token budget.

**Parsing**

- Split at the last `</think>`. If it is missing, the answer is incorrect.
- Prediction = contents of the last `\boxed{...}` after it; otherwise the last number after it.
- Normalize by removing commas, `$`, `%` and a trailing `.0`, then compare as numbers.
- Cover the parser with at least 20 hand-written test cases.

**Job summary** in DONE.json: n, hit-limit rate, accuracy, mean tokens generated, tokens per second. The runner uses measured tokens per second to update later job estimates.

**Pilot** (P0): base and edit, dev50, F16. It must show the edit cutting the hit-limit rate by at least 20 points at 8,192 tokens. If not, raise the budget to 16,384, bump `design_version` and recompute section 10.

## 9. Stage 5: weight analysis and statistics

**Weight analysis** runs on CPU (laptop is fine; tensors are processed one at a time, so 8 GB of RAM is enough). All comparisons happen inside GGUF files, so tensor names always match.

For each condition v other than base, each level L and each tensor t:

```latex
\Delta_L(t) = \mathrm{deq}(v, L, t) - \mathrm{deq}(\mathrm{base}, L, t), \qquad \Delta = \Delta_{F16}
```

Dequantize with `gguf.quants.dequantize`. Per tensor, record:

- Norms of the change before and after quantization, and survival = norm after / norm before.
- Cosine similarity between the two changes.
- Share of changed F16 entries whose quantized value also changed.
- Median of |Δ| divided by the nominal block step: the base block's largest |w| / (2^(b−1) − 1) for the tensor's quant type at b bits.

Write one Parquet file per (condition, level) to `results/weights/`.

**Statistics**

- Effect E(v, L) = metric(v, L) − metric(base, L), for hit-limit rate (primary) and accuracy.
- Retention R(v, L) = E(v, L) / E(v, F16).
- 95% confidence intervals by paired bootstrap over problems, 10,000 resamples, with the same problem ids drawn for v and base at both levels.
- Report seed 0 for the full grid, and mean and range across seeds where all three exist.
- **Floor rule:** if base at a level scores under 2% accuracy and over 98% hit-limit, mark that level "floor" and skip R.

**Figures** (`python -m wsq report` rebuilds them from Hub data)

1. Retention against bits per weight, one line per method, with CI bands.
2. Per-tensor survival against median |Δ|/step, coloured by method.
3. Table of the edited tensors' types at each level.

## 10. Compute budget

Evaluation estimates assume about 1,000 generated tokens per second per T4 with 16 parallel requests, and about 6,000 tokens per problem on average (base uses nearly all 8,192). Training estimates assume the L4's fast bf16 path. All are guesses; the pilot and the first training run measure them.

| Work | Where | Count | Each | Total GPU time |
| --- | --- | --- | --- | --- |
| Pilot (base and edit, dev50, F16) | Kaggle T4 | 2 | 5 min | 0.2 h |
| llama.cpp builds (cached afterwards) | Both | 2 platforms | 10 min | 0.3 h |
| LoRA training (calibration + 2 seeds) | Colab L4 | 3 | 5–15 min | 0.5 h |
| Full fine-tune training (calibration + 2 seeds) | Colab L4 | 3 | 10–25 min | 1 h |
| Effect-matching evals (dev50, F16) | Colab L4 | 10 | 4 min | 0.7 h |
| Conversion and quantization | Kaggle T4 | ~48 files | 1–2 min | 1 h |
| Main evals (test150): 8 conditions × 6 levels | Kaggle T4 | 48 | 15 min | 12 h |
| **Colab Pro L4 total** | | | | **~2.5–4 h, about 4–6 compute units** |
| **Kaggle T4 total** | | | | **~13.5 h (range 9–30 h)** |
| Weight analysis | Laptop CPU | ~42 pairs | 1–2 min | 1 h of CPU |

A free Kaggle account's weekly GPU allowance (about 30 hours, plus 15 once Colab Pro is linked) covers the evaluation in one to two weeks, and a two-T4 session roughly halves it. If Kaggle quota runs short, eval jobs can also run on Colab Pro while you are at the laptop; users have measured about 1.1 compute units per T4 hour and 1.5 per L4 hour.

**If the pilot projects more than 30 h**, cut in this order:

1. Run seeds 1–2 only at F16, Q4_K_pure and Q3_K_pure (saves about 3 h).
2. Shrink the test set from 150 to 100 problems.
3. Drop Q6_K_pure.

## 11. Four-week plan

1. **Week 1 (18–24 Nov): set-up and pilot.** Repo, CPU smoke test, Hugging Face and Kaggle set-up, llama.cpp cache on both platforms, edit check, pilot.
   - Smoke test passes, including a kill-and-resume run.
   - Edit check finds exactly the two expected tensors.
   - Pilot on a Kaggle T4 shows the edit cutting hit-limit rate by at least 20 points; measured tokens per second recorded and section 10 updated.
   - A short Colab Pro L4 session confirms the fast training path loads, or logs that the fallback is in use.
2. **Week 2 (25 Nov–1 Dec): variants.** Training data, then LoRA and full fine-tune calibration and seeds 1–2, all on the Colab Pro L4.
   - Chosen checkpoints are within 5 points of the edit's effect on dev50.
   - Exports pass the name, shape and dtype checks.
   - Quantization manifests and section 7 checks pass for every condition.
3. **Week 3 (2–8 Dec): main grid.** P2–P5 jobs as Kaggle batch runs; weight analysis on the laptop as files appear.
   - All P2–P4 jobs are DONE and `wsq status` shows no stale leases.
4. **Week 4 (9–15 Dec): analysis and report.** Statistics, figures, write-up.
   - `python -m wsq report` rebuilds every figure and table from Hub data in one command.

## 12. Risks and fallbacks

| Risk | Sign | Fallback |
| --- | --- | --- |
| Qwen3.5's Gated DeltaNet kernels fail | Triton compile errors (on a T4: `LLVM ERROR: Unsupported rounding mode`) | Train only on the L4. If the fast kernels still fail there, uninstall `flash-linear-attention` and `causal-conv1d` and use the PyTorch fallback. Run all inference in llama.cpp, not vLLM |
| No L4 available | Colab assigns a T4 or A100 | Wait and retry. If L4s stay unavailable, move all training (every seed) to the A100 and bump `design_version`; never train on a T4 |
| Colab session drops during training | Laptop slept or idle timeout | Training resumes from the last 25-step checkpoint |
| Training too slow on the fallback path | Over 1 h per LoRA run | Cut maximum length to 512 tokens and use fewer steps |
| F16 GGUF misbehaves | Nonsense from an F16 file | Check it against the HF model on five prompts; if it fails, convert with `--outtype bf16` and run F16-level evals on the L4 |
| Edit effect too small at 8,192 tokens | Pilot drop under 20 points | Raise the budget to 16,384 and redo section 10 |
| Thinking mode not on | Rendered prompt lacks the thinking opener | Pass `enable_thinking=True` explicitly; the section 6 assertion catches it |
| Kaggle secret missing in a batch run | Hugging Face auth error in cell 1 | Private Kaggle dataset holding the token |
| Hub rate limits | HTTP 429 | Back off; one commit per job per 5 minutes |
| 3-bit level breaks every model | Base accuracy near zero | Floor rule (section 9) |
| llama.cpp stop fields differ from expected | `hit_limit` missing or always false | Read field names from a real response; unit test |
| Two workers take the same job | Two live leases | Lease check plus identical outputs make it harmless |

## 13. Sources

- [Project proposal](https://claude.ai/artifact/FtsvsEapHtXnT1jQvdXPs6)
- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi et al., 2026): the edit, its sampling settings and results
- [Qwen/Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B): hybrid Gated DeltaNet architecture
- [Qwen3.5 fails on T4: BF16 Gated DeltaNet kernel](https://github.com/strands-labs/strands-decider/issues/113)
- [openai/gsm8k dataset card](https://huggingface.co/datasets/openai/gsm8k/blame/main/README.md)
- [Kaggle CLI: kernels push and accelerators](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Kaggle batch sessions (Save & Run All)](https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html)
- [Colab VS Code extension user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
- [Colab VS Code extension: no access to local files](https://github.com/googlecolab/colab-vscode/issues/210)
- [Colab plan limits: background execution is Pro+ only](https://fastero.com/blog/google-colab-limits)
- [Colab GPUs and measured compute units per hour](https://fastero.com/blog/google-colab-gpu)
