# Notebooks

Colab/Kaggle-friendly notebooks to run without paid resources. Suggested set:

1) `01_clean_enron.ipynb` — download Enron (or other) dataset, extract, clean via `app/cleaner.py`, and save to `data/processed/enron_clean.jsonl`. Keeps data size manageable (sample or cap rows) for free GPUs.
2) `02_label_multitask.ipynb` — generate labels for multiple tasks (summarize, reply, rewrite, phishing, spam, categorize). Uses local/hosted teacher (e.g., Ollama `qwen2.5:7b-instruct` or free OpenAI-compatible endpoint). Enforces 5-line or task-specific schemas and caps `MAX_ITEMS` to stay within free limits.
3) `03_train_multitask_qlora.ipynb` — QLoRA fine-tune on free GPUs (Kaggle/Colab). Defaults: 4-bit, seq len 1024–1536, grad accumulation 16–32, batch size 1, cosine LR, 1–2 epochs. Saves adapter + tokenizer to `/kaggle/working/train/output` or `/content/train/output`.
4) `04_eval_and_infer.ipynb` — quick evaluation plus demo for each task through `LocalSummarizer` (or future multi-task router). Includes speed tips (smaller `max_new_tokens`, CPU fallback) and sample inputs.

You can create these notebooks in-place on Kaggle/Colab, then `File -> Download .ipynb` to commit them back. Keep datasets small (e.g., sample 2k–5k rows) to fit free tiers.
