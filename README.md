# DeepSeek-R1-Presentation
# Margot Zhao
# 3/26/2025

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## Introduction 
Good afternoon everyone! Today I'll be presenting the paper "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" from DeepSeek-AI. This paper introduces significant advancements in developing reasoning capabilities in large language models through reinforcement learning techniques.

## Overview 
### Problem Statement
Current LLMs face challenges in complex reasoning tasks despite their impressive capabilities. While models like OpenAI's o1 have made progress through inference-time scaling with longer Chain-of-Thought processes, the wider research community lacks effective methods to achieve comparable reasoning performance.

### Research Question
The core question this paper addresses is: Can pure reinforcement learning (RL) be used to enhance reasoning capabilities in LLMs without relying heavily on supervised fine-tuning (SFT)?

### Approach
The researchers took a novel approach by:
1. Applying reinforcement learning directly to base models without supervised fine-tuning as an initial step (DeepSeek-R1-Zero)
2. Developing a multi-stage training pipeline incorporating cold-start data and iterative RL fine-tuning (DeepSeek-R1)
3. Distilling reasoning capabilities from larger models to smaller dense models

### Contributions
The paper makes three key contributions:
1. Demonstrating that reasoning capabilities can be incentivized purely through RL (DeepSeek-R1-Zero)
2. Introducing a viable pipeline combining limited supervised data with extensive RL
3. Showing that reasoning patterns from larger models can be effectively distilled to smaller ones

## Architecture Overview 
### DeepSeek-R1-Zero Architecture
```
Algorithm: Group Relative Policy Optimization (GRPO)
Base Model: DeepSeek-V3-Base
Training Process:
1. Initialize with base model without SFT
2. For each reasoning question q:
   - Sample multiple outputs {o₁, o₂, ..., oₖ} from current policy
   - Compute rewards {r₁, r₂, ..., rₖ} using rule-based evaluation
   - Calculate advantages A_i = (r_i - mean(rewards))/std(rewards)
   - Update policy to maximize GRPO objective:
     J_GRPO(θ) = E[min(π_θ(o|q)/π_old(o|q) × A, clip(π_θ(o|q)/π_old(o|q), 1-ε, 1+ε) × A)]
3. Output format: <think>reasoning process</think><answer>final answer</answer>
```

### DeepSeek-R1 Architecture
```
Multi-Stage Pipeline:
Stage 1: Cold-Start Data Collection
   - Collect thousands of high-quality CoT examples
   - Fine-tune DeepSeek-V3-Base on this data

Stage 2: Reasoning-Oriented RL
   - Apply GRPO framework with rule-based rewards
   - Add language consistency reward to prevent language mixing

Stage 3: Rejection Sampling and SFT
   - Generate diverse reasoning data through rejection sampling
   - Incorporate non-reasoning data (writing, factual QA, etc.)
   - Fine-tune with combined dataset (~800k samples)

Stage 4: RL for All Scenarios
   - Apply RL to improve helpfulness and harmlessness
   - Maintain reasoning capabilities
```

## Key Results 
DeepSeek-R1 achieves impressive performance across various benchmarks:
- AIME 2024: 79.8% (outperforming OpenAI-o1-1217's 79.2%)
- MATH-500: 97.3% (comparable to OpenAI-o1-1217's 96.4%)
- Codeforces: 96.3 percentile rating (similar to OpenAI-o1-1217's 96.6)
- MMLU: 90.8% (slightly below OpenAI-o1-1217's 91.8%)
- GPQA Diamond: 71.5% (below OpenAI-o1-1217's 75.7%)

The distilled models also show strong performance:
- DeepSeek-R1-Distill-Qwen-7B: 55.5% on AIME 2024 (outperforming QwQ-32B-Preview)
- DeepSeek-R1-Distill-Qwen-32B: 72.6% on AIME 2024, 94.3% on MATH-500

## Critical Analysis 
### Limitations
1. Language mixing issues - DeepSeek-R1-Zero struggled with mixing languages in outputs
2. Readability challenges - outputs required additional structuring to be human-friendly
3. Software engineering performance - not significantly improved over DeepSeek-V3 due to evaluation time constraints
4. Prompt sensitivity - DeepSeek-R1 performs better with zero-shot than few-shot prompting
5. Non-reasoning tasks may show reduced performance compared to specialized models

### Unsuccessful Attempts
The authors transparently discuss failed approaches:
1. Process Reward Models (PRMs) - challenges in defining fine-grained steps and reward hacking
2. Monte Carlo Tree Search (MCTS) - exponentially large search space and difficulty training value models

## Impacts and Future Directions 
This work impacts the AI landscape by:
1. Proving reinforcement learning alone can significantly enhance reasoning capabilities
2. Demonstrating effective distillation of reasoning abilities to smaller models
3. Providing open-source models for community research
4. Establishing a framework for future improvements in language model reasoning

Future work directions include:
1. Enhancing general capabilities (function calling, multi-turn conversation)
2. Addressing language mixing issues
3. Improving software engineering capabilities
4. Optimizing prompting techniques

## Questions for the Audience
### Question 1
The paper showed that RL without supervised fine-tuning can develop remarkable reasoning capabilities. What implications do you think this has for the future development of AI systems that can perform complex, multi-step reasoning?

### Question 2
The authors observed an "aha moment" where DeepSeek-R1-Zero spontaneously developed self-verification and reflection capabilities. What might this tell us about the emergence of cognitive-like abilities in large language models?

## Resources and References
- GitHub Repository: https://github.com/deepseek-ai/DeepSeek-R1
- Paper: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (Jan 2025)
- Model API: Available through DeepSeek's API services
- Related Work: OpenAI's o1 series, GRPO framework paper, Math-Shepherd

*Note: I'll conclude by taking questions from the audience, and direct them to our repository for further information, code examples, and detailed documentation.*
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}
