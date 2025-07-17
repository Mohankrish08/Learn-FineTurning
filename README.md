# Fine-Tuning and Supervised Fine-Tuning

## What is Fine Turning?

It is a process of adapting a pre-trained Large Language Model (LLM) to specific domain or task by training of new data.
Instead of training a model from Scratch, fine turning uses the existing knowledge of the base model and utilize the custom use case data.

Eg: Legal document analysis, Chatbots and Medical Q&A.

### Why Fine Turning?

- Improve performance of domain specific tasks.
- Requires less compute than training an actual model.
- Retrain general knowledge understanding.

---

## Types of Fine Tuning:

| **Type**                                              | **Description**                                                             | **Parameters Updated**                    | **Data Requirement**              | **Best For**                                          |
| ----------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------- | ----------------------------------------------------- |
| **Full Fine-Tuning**                                  | Updates **all model weights** on new data                                   | All parameters                            | Large labeled/unlabeled dataset   | Domain specialization, when high compute is available |
| **Supervised Fine-Tuning (SFT)**                      | Trains on **labeled input → output (instruction-response) pairs**           | All or partial (with PEFT)                | Labeled instruction-response data | Instruction following, Q\&A, chatbots                 |
| **PEFT (Parameter-Efficient Fine-Tuning)**            | Updates **only a small subset of parameters** while freezing the base model | Few (adapters, LoRA matrices, bias terms) | Small to medium datasets          | Resource-efficient fine-tuning                        |
| ➝ **LoRA**                                            | Adds **low-rank adapter matrices** to attention layers (`q_proj`, `v_proj`) | Only adapter weights                      | Small datasets                    | Instruction tuning on limited GPUs                    |
| ➝ **QLoRA**                                           | Quantized LoRA (4-bit base model + LoRA adapters)                           | Only adapter weights                      | Small datasets                    | Large LLM fine-tuning on single GPU                   |
| ➝ **Prefix / Prompt Tuning**                          | Learns **special tokens or prompts**, base model frozen                     | Only soft prompt vectors                  | Very small datasets               | Few-shot learning, quick adaptations                  |
| ➝ **BitFit**                                          | Updates only **bias terms**                                                 | Bias parameters only                      | Very small datasets               | Extremely lightweight fine-tuning                     |
| **Continued Pretraining**                             | Continues pretraining on **domain-specific unlabeled data**                 | All parameters                            | Large unlabeled dataset           | Domain adaptation before fine-tuning                  |
| **RLHF (Reinforcement Learning with Human Feedback)** | Aligns model with human preferences via a reward model + PPO                | Base model + reward model                 | Human preference data             | Human-aligned chatbots (e.g., ChatGPT)                |
| **DPO (Direct Preference Optimization)**              | Simplified RLHF, directly optimizes on human preference pairs               | Base model parameters                     | Human preference data             | Chatbot alignment with less complexity                |


---
## SFT + PEFT

[Indian IPC FT by Lora](https://github.com/Mohankrish08/Learn-FineTurning/blob/main/Supervised%20FineTurning/PEFT/Lora/Indian_IPC_FT.ipynb)


## What is Supervised Fine-Turning?

It is type of Fine turning model learns from the labeled data (input --> output pairs) in supervised manner.

### When to use SFT?

- When you have instruction-response pairs or conversational data.
- When you want the model to follow specific instructions consistently.
- When preparing a base model for further alignment steps like DPO or RLHF.

### What is PEFT?

- It is technique that small set of model parameters are trained while majority of parameters will be frozon.
- It dratically reduce the computation, memory usage and training time.

### What is Lora?

- One of the most popular `PEFT` Technique. 
- It works by adding low-rank adapter layers (r rank matrices) to specific parts of the model (mainly attention layers such as `q_proj and v_proj`)
- During the Fine-Turning, only these small adapter layers will be updated, while the base remain frozen.

#### Lora for SFT

- Efficient
- Faster Traning
- Retrain Base knowledge
- Easy to Merge and share

#### Comparision: Full FT vs Lora:

| Aspect                   | Full Fine-Tuning                | LoRA (PEFT)                         |
| ------------------------ | ------------------------------- | ----------------------------------- |
| **Parameters Updated**   | All model parameters            | Only adapter weights (`r` rank)     |
| **Compute Cost**         | Very High (needs A100/TPU)      | Very Low (can run on 1 GPU/Colab)   |
| **Training Speed**       | Slow                            | Fast                                |
| **Preserves Base Model** | No (weights overwritten)        | Yes (base model stays frozen)       |
| **Best For**             | Domain-specific full adaptation | Instruction tuning / small datasets |
