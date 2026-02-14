from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from configs.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


EXPECTED_PREFIXES = [
    "1) Purpose:",
    "2) Key details:",
    "3) Action needed:",
    "4) Deadline/Date:",
    "5) Important context/links:",
]


def _build_messages(subject: str, body: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(subject=subject or "(no subject)", body=body),
        },
    ]


def _validate_summary(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) != 5:
        return False
    for line, prefix in zip(lines, EXPECTED_PREFIXES):
        if not line.startswith(prefix):
            return False
    return True


def _coerce_summary(text: str) -> str:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    # Pad or trim to 5 lines with placeholders
    coerced: List[str] = []
    for idx, prefix in enumerate(EXPECTED_PREFIXES):
        if idx < len(lines) and lines[idx].startswith(prefix):
            coerced.append(lines[idx])
        elif idx < len(lines):
            coerced.append(f"{prefix} {lines[idx]}")
        else:
            coerced.append(f"{prefix} N/A")
    return "\n".join(coerced[:5])


class LocalSummarizer:
    def __init__(
        self,
        model_name: str = "Qwen3-4B-Instruct-2507",
        adapter_path: str = "train/output",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path if Path(adapter_path).exists() else model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()

    def _generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def summarize(self, subject: str, body: str) -> str:
        messages = _build_messages(subject, body)
        draft = self._generate(messages)
        if _validate_summary(draft):
            return draft

        correction_messages = messages + [
            {
                "role": "user",
                "content": "Output EXACTLY 5 lines following the schema. Fix formatting now without adding explanation.",
            }
        ]
        retry = self._generate(correction_messages)
        if _validate_summary(retry):
            return retry

        return _coerce_summary(retry)
