# Email Agent – Local 5-Line Summarizer

Build a local email summarizer distilled from a teacher model (Llama-3.3-70B-Instruct) into a student (Qwen3-4B-Instruct-2507) fine-tuned with QLoRA. Summaries always follow the fixed 5-line schema.

## Setup (Ubuntu)
- Python 3.10+ recommended.
- Create env and install deps:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
- Copy env template and edit secrets:
```
cp .env.example .env
```

## GitHub & Free GPU Workflow (Kaggle/Colab)
- Initialize git locally and push to your GitHub repo (exclude data and secrets; `.env`, `data/*`, and `train/output/*` are already ignored).
- On Kaggle/Colab: clone the repo, install deps (`pip install -r requirements.txt`), and use the notebooks in `notebooks/` for data prep, labeling, training, and inference on free GPUs.
- Cache models/tokenizers to writable paths: Kaggle (`/kaggle/temp`), Colab (`/root/.cache/huggingface`).
- Keep datasets small for free tiers: sample 2k–5k emails; limit sequence length to 1024–1536; use grad accumulation (16–32) with batch size 1.

## Dataset: Enron Maildir
1) Download (tarball ~423MB):
```
bash scripts/01_download_enron.sh
tar -xzf data/raw/enron_mail_20110402.tgz -C data/raw
```
2) Parse + clean to JSONL:
```
python scripts/02_parse_and_clean_enron.py
```
Output: `data/processed/enron_clean.jsonl` with fields `source, path, subject, clean_text, urls`.

Cleaning steps (app/cleaner.py): HTML→text, quoted-history removal, signature heuristics, whitespace normalization, URL extraction, thread truncation.

## Label with Teacher (OpenAI-compatible API)
Environment (.env): `TEACHER_API_BASE`, `TEACHER_API_KEY`, `TEACHER_MODEL` (default meta-llama/Llama-3.3-70B-Instruct), `MAX_LABEL_ITEMS`, `SLEEP_SEC`.

Run labeling:
```
python scripts/03_label_with_teacher.py
```
Output: `data/labeled/enron_labeled.jsonl` in chat-style messages. The script validates the 5-line format, retries once, then skips invalid samples.

## Train Student with QLoRA (GTX 1650 Ti 4GB safe defaults)
Model: `Qwen3-4B-Instruct-2507` (default arg uses this HF ID; override if needed).

Recommended command:
```
python train/04_train_qwen3_4b_qlora.py \
  --data_path data/labeled/enron_labeled.jsonl \
  --output_dir train/output \
  --model_name Qwen3-4B-Instruct-2507 \
  --max_seq_length 2048 \
  --grad_accum_steps 16 \
  --learning_rate 2e-4 \
  --num_epochs 2
```
Memory tips:
- If you see CUDA OOM: set `--max_seq_length 1536` (or 1024), and/or `--grad_accum_steps 32`.
- Close browsers/GPUs hogs; ensure `TORCH_CUDA_ARCH_LIST` unset unless needed.
- Training uses NF4 4-bit, paged_adamw_8bit, gradient checkpointing, batch size 1.

Resume training:
```
python train/04_train_qwen3_4b_qlora.py --resume_from train/output/checkpoint-xxxx
```

## Local Inference
After training, run a sample email:
```
python train/05_infer_local.py --adapter_path train/output --subject "Weekly update" --text "We shipped feature X yesterday..."
```

CLI wrapper (cleans text first): defaults use your trained adapter in `train/output`.
```
# Paste interactively (no temp file):
python -m app.cli
# paste email body, press Ctrl-D (Ctrl-Z on Windows)

# Pipe from clipboard:

# Inline text:
python -m app.cli --text "Please sign the doc by Friday"

# File input:
python -m app.cli --file sample_email.txt
```
You can override paths via env vars: `SUMMARY_MODEL_NAME` and `SUMMARY_ADAPTER_PATH`.
Output always 5 lines:
```
1) Purpose: ...
2) Key details: ...
3) Action needed: ...
4) Deadline/Date: ...
5) Important context/links: ...
```

## Pipeline Overview
- Data: `scripts/01_download_enron.sh` → extract → `scripts/02_parse_and_clean_enron.py` → `data/processed/enron_clean.jsonl`.
- Label: `scripts/03_label_with_teacher.py` → `data/labeled/enron_labeled.jsonl` (teacher API, format enforcement).
- Train: `train/04_train_qwen3_4b_qlora.py` → LoRA adapter in `train/output`.
- Infer: `train/05_infer_local.py` or `python -m app.cli summarize ...` (validator retries or coerces to 5 lines).

## Extending to Multi-Task (planned)
- Add task-specific prompts for: summarize, generate reply, rewrite (professional/friendly/short), phishing detection, spam classification, auto-categorize, smart search (with embeddings), and attachment metadata analysis.
- Use `notebooks/02_label_multitask.ipynb` to bootstrap synthetic labels with a local/hosted teacher (e.g., Ollama `qwen2.5:7b-instruct`).
- Train a small QLoRA adapter on the multi-task mix (`notebooks/03_train_multitask_qlora.ipynb`) for free Kaggle/Colab GPUs.

## Troubleshooting
- CUDA OOM: lower `--max_seq_length` (2048→1536→1024), increase `--grad_accum_steps`, or set `CUDA_VISIBLE_DEVICES=""` to force CPU (slow). Restart notebook/kernel to release VRAM.
- Slow inference: use `--max_new_tokens 128` (edit summarizer), ensure `torch.set_num_threads` reasonable, close other GPU apps.
- CPU fallback: if no CUDA, the summarizer loads in float32 on CPU automatically (much slower). Consider smaller models for experimentation.
- Invalid summaries: summarizer auto-retries; if still wrong, it coerces format. Check `configs/prompt_templates.py` to tighten phrasing.

## Provider Stub
`app/provider_stub.py` exposes `fetch_emails_imap()` returning sample messages and a note on future Gmail integration. Replace with real IMAP/Gmail API later.
