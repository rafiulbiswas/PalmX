# PalmX Arabic Islamic QA Fine-tuning

A sophisticated fine-tuning system for Arabic Islamic culture question-answering tasks using Qwen2.5-7B-Instruct with LoRA (Low-Rank Adaptation) for the [PalmX 2025 competition](https://palmx.dlnlp.ai/).

## 🎯 Overview

This project fine-tunes the Qwen2.5-7B-Instruct model to excel at Arabic Islamic knowledge multiple-choice questions. The system uses advanced parameter-efficient fine-tuning techniques and cultural-aware prompt engineering to achieve superior performance on Islamic culture and knowledge tasks.

**Target Performance**: Beat the 69.5% accuracy baseline set by NileChat-3B

## 🚀 Features

- **Parameter-Efficient Fine-tuning**: Uses LoRA with optimized configurations
- **Memory Optimization**: 4-bit quantization with BitsAndBytes
- **Cultural Adaptation**: Arabic-first prompt engineering with Islamic context
- **Advanced Training**: Cosine scheduling, gradient clipping, early stopping
- **Comprehensive Evaluation**: Baseline-compatible scoring with detailed error analysis
- **Production-Ready**: Robust error handling, logging, and checkpoint management

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA A100 or equivalent (recommended)
- **RAM**: 16GB+ system memory
- **VRAM**: 24GB+ GPU memory
- **Storage**: 50GB+ free space

### Software
```bash
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
bitsandbytes>=0.39.0

# Additional requirements
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd palmx-arabic-qa
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
# OR in Google Colab:
!pip install bitsandbytes transformers datasets peft torch
```

3. **Use HuggingFace Model**
```
https://huggingface.co/rafiulbiswas/qwen2.5-7b-arabic-culture-qa

```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from palmx_finetuner import ImprovedQwenFineTuner

# Initialize the fine-tuner
ft = ImprovedQwenFineTuner(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    token="your_hf_token"
)

# Setup model and LoRA configuration
ft.setup_model_and_tokenizer()
ft.setup_improved_lora_config(r=32, lora_alpha=64)

# Load and prepare data
train_data, eval_data = ft.load_and_prepare_data("subtask2")

# Fine-tune the model
trainer = ft.fine_tune_improved(train_data, eval_data)

# Evaluate performance
results, accuracy = ft.evaluate_with_baseline_format(eval_data)
print(f"Accuracy: {accuracy:.2f}%")
```

### Running the Complete Pipeline

```python
# Execute the main fine-tuning pipeline
if __name__ == "__main__":
    main_improved_finetune()
```

## 📊 Dataset Information

The system supports two PalmX 2025 competition subtasks:

- **Subtask 1**: [`UBC-NLP/palmx_2025_subtask1_culture`](https://huggingface.co/datasets/UBC-NLP/palmx_2025_subtask1_culture) - General Arabic culture
- **Subtask 2**: [`UBC-NLP/palmx_2025_subtask2_islamic`](https://huggingface.co/datasets/UBC-NLP/palmx_2025_subtask2_islamic) - Islamic knowledge

**Competition**: [PalmX 2025](https://palmx.dlnlp.ai/)

### Data Format
Each example contains:
- `question`: Arabic question text
- `A`, `B`, `C`, `D`: Multiple choice options
- `answer`: Correct answer (A, B, C, or D)
- `id`: Unique identifier

## ⚙️ Configuration

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=32,                    # Rank (higher = more capacity)
    lora_alpha=64,          # Scaling factor
    lora_dropout=0.05,      # Regularization
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
        "gate_proj", "up_proj", "down_proj",        # MLP
        "embed_tokens", "lm_head"                   # Embeddings
    ]
)
```

### Training Parameters
```python
training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=200,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,
    gradient_checkpointing=True
)

## 🔧 Advanced Usage

### Custom Prompt Engineering
```python
def custom_prompt_format(example):
    return f"""<|im_start|>system
أنت خبير في الثقافة الإسلامية. أجب على السؤال متعدد الخيارات.
<|im_end|>
<|im_start|>user
{example['question']}
A. {example['A']}
B. {example['B']}
C. {example['C']}
D. {example['D']}
<|im_end|>
<|im_start|>assistant
{example['answer']}<|im_end|>"""
```

### Memory Management
```python
# Clear GPU cache regularly
torch.cuda.empty_cache()
gc.collect()

# Monitor GPU memory
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

### Error Analysis
```python
# Analyze model errors
error_analysis = ft.analyze_errors(results, eval_data)
print(f"Error distribution: {error_analysis['by_answer']}")
print(f"Difficult questions: {len(error_analysis['difficult_questions'])}")
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the base Qwen2.5-7B-Instruct model
- **UBC-NLP** for the PalmX 2025 datasets
- **HuggingFace** for transformers and datasets libraries
- **Microsoft** for the LoRA technique implementation

---

**Note**: This system is designed for research and educational purposes. Ensure you have proper authorization for model training and dataset usage.
