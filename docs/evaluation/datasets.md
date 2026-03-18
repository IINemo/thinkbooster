# Datasets

All datasets are available in our HuggingFace collection: [test-time-compute](https://huggingface.co/test-time-compute)

## Supported Datasets

| Dataset | Config | Size | Domain | Description |
|---------|--------|------|--------|-------------|
| **MATH-500** | `dataset/math_500` | 500 | Math | Competition-level math (OpenAI "Let's Verify Step by Step" subset) |
| **OlympiadBench** | `dataset/olympiadbench` | 675 | Math | Olympiad-level math problems |
| **GaoKao 2023 En** | `dataset/gaokao2023en` | 385 | Math | Chinese college entrance exam (English) |
| **Minerva Math** | `dataset/minerva_math` | 272 | Math | Minerva math reasoning problems |
| **AIME 2025** | `dataset/aime_2025` | 30 | Math | American Invitational Mathematics Examination |
| **AIME 2024** | — | 30 | Math | AIME 2024 edition |
| **AMC 2023** | — | — | Math | AMC competition problems |
| **GPQA Diamond** | `dataset/gpqa_diamond` | — | Science | Graduate-level science QA |
| **HumanEval+** | `dataset/human_eval_plus` | 164 | Code | Python function synthesis (EvalPlus) |
| **MBPP+** | `dataset/mbpp_plus` | 378 | Code | Mostly Basic Python Problems (EvalPlus) |
| **Game of 24** | `dataset/game24` | — | Math | Reach 24 using 4 numbers |
| **GSM8K** | `dataset/gsm8k` | 1,319 | Math | Grade school math word problems |

---

## Dataset Details

### AIME 2024 / AIME 2025

American Invitational Mathematics Examination problems.

- **Source**: [test-time-compute/aime_2024](https://huggingface.co/datasets/test-time-compute/aime_2024), [test-time-compute/aime_2025](https://huggingface.co/datasets/test-time-compute/aime_2025)
- **Answer format**: Integer between 0 and 999
- **Difficulty**: Very hard (competition-level)
- **Widely used** in test-time compute papers

```yaml
dataset:
  name: test-time-compute/aime_2025
  split: test
  question_field: problem
  answer_field: answer
```

### MATH-500

Subset of MATH dataset created by OpenAI for the "Let's Verify Step by Step" paper.

- **Source**: Original MATH dataset with OpenAI's subset selection
- **Answer format**: Numeric or mathematical expression
- **Requires**: Math parsing for exact match comparison
- **Note**: Full MATH dataset (released 2021) may be saturated

---

## Dataset Loading Example

```python
from datasets import load_dataset

# AIME 2025
dataset = load_dataset("test-time-compute/aime_2025", split="test")

# Access fields
for sample in dataset:
    question = sample["problem"]
    answer = sample["answer"]  # Integer string
```

---

## Notes

- Some strategies were originally developed for code reasoning tasks and evaluated on code validity, not math.
- Some papers used custom/crafted datasets not publicly available.
- We focus on publicly available math reasoning benchmarks for reproducibility.
