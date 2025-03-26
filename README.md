# DeepSeek-R1-Presentation
# Margot Zhao
# 3/26/2025

This repository contains materials for a presentation of the paper "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv:2501.12948v1, Jan 2025).

## Paper Summary

DeepSeek-R1 represents a breakthrough in enhancing reasoning capabilities of Large Language Models (LLMs) through reinforcement learning. The paper introduces two main models:

1. **DeepSeek-R1-Zero**: A model trained via large-scale reinforcement learning without supervised fine-tuning, demonstrating remarkable reasoning capabilities.

2. **DeepSeek-R1**: A model that incorporates multi-stage training and cold-start data before reinforcement learning.

The research also demonstrates successful distillation of reasoning capabilities to smaller dense models (1.5B to 70B parameters).

## Key Contributions

- First open research to validate that reasoning capabilities can be incentivized purely through RL without SFT.
- Introduction of a training pipeline combining RL and SFT stages that produces state-of-the-art reasoning performance.
- Demonstration that reasoning patterns from larger models can be distilled into smaller ones.
- Open-sourcing of DeepSeek-R1-Zero, DeepSeek-R1, and six dense distilled models.

## Performance Highlights

DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks:

| Benchmark | DeepSeek-R1 | OpenAI-o1-1217 |
|-----------|-------------|----------------|
| AIME 2024 (Pass@1) | 79.8% | 79.2% |
| Codeforces (Percentile) | 96.3% | 96.6% |
| GPQA Diamond (Pass@1) | 71.5% | 75.7% |
| MATH-500 (Pass@1) | 97.3% | 96.4% |
| MMLU (Pass@1) | 90.8% | 91.8% |
| SWE-bench Verified (Resolved) | 49.2% | 48.9% |
![image](https://github.com/user-attachments/assets/5bded521-b00a-45d5-9ea1-386c7bcb18a0)
![image](https://github.com/user-attachments/assets/cdb4ee9f-655d-4768-b4dc-dca0c900bc1b)
![image](https://github.com/user-attachments/assets/b7107916-01ce-4896-b5a1-2745ecc89dde)




## Technical Approach

### DeepSeek-R1-Zero
- Direct RL on the base model (DeepSeek-V3-Base) without SFT
- Group Relative Policy Optimization (GRPO) as the RL framework
- Rule-based rewards for accuracy and format
- Natural emergence of sophisticated reasoning behaviors

### DeepSeek-R1
1. Cold-start with high-quality CoT data
2. Reasoning-oriented RL
3. Rejection sampling and supervised fine-tuning
4. Final RL for all scenarios

### Distillation
- Transfer of reasoning patterns to smaller dense models
- Models from 1.5B to 70B parameters based on Qwen and Llama

## Demos and Examples

This repository includes a [Jupyter notebook](./demos/reasoning_examples.ipynb) demonstrating examples of the reasoning capabilities of DeepSeek-R1, including:

1. Mathematical problem-solving
2. Coding challenges
3. Scientific reasoning tasks
4. Emergent behavior examples (self-verification, reflection)

## Limitations and Future Work

- General capability gaps compared to DeepSeek-V3 in function calling, multi-turn interactions, and complex role-playing
- Language mixing issues when handling queries in languages other than English/Chinese
- Sensitivity to prompting (performs best with zero-shot)
- Limited improvement in software engineering tasks

## Resources

- [Original Paper on arXiv](https://arxiv.org/abs/2501.12948)
- [DeepSeek-AI Official GitHub](https://github.com/deepseek-ai)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) - The RL algorithm used
- [OpenAI's Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) - Related research on reasoning
- [Llama 3 Paper](https://arxiv.org/abs/2407.21783) - Relevant foundation model used in distillation

## Citation

```bibtex
@article{deepseek2025r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}
