# DeepSeek-R1-Presentation
# Margot Zhao
# 3/26/2025

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## Overview 
### Problem Statement
DeepSeek-R1 represents a breakthrough in enhancing reasoning capabilities of Large Language Models (LLMs) through reinforcement learning. The paper introduces two main models:

1. **DeepSeek-R1-Zero**: A model trained via large-scale reinforcement learning without supervised fine-tuning, demonstrating remarkable reasoning capabilities.

2. **DeepSeek-R1**: A model that incorporates multi-stage training and cold-start data before reinforcement learning.

The research also demonstrates successful distillation of reasoning capabilities to smaller dense models (1.5B to 70B parameters).

![benchmark](https://github.com/user-attachments/assets/d386014a-98cc-43fe-b29b-65b367939605)

- AIME 2024 (Pass@1): American Invitational Mathematics Examination 
- Codeforces (Percentile): is a competitive programming platform. 
- GPQA Diamond (Pass@1): Graduate-level Google-Proof Q&A) tests graduate-level knowledge across physics, chemistry, biology, and more. 
- MATH-500 (Pass@1): A collection of 500 challenging math problems spanning algebra, geometry, calculus, and more. 
- MMLU (Pass@1): Massive Multitask Language Understanding evaluates knowledge across 57 subjects including science, humanities, law, and medicine.
- SWE-bench Verified (Resolved): Software Engineering benchmark that tests the model's ability to fix real-world software bugs in open-source projects.

### Research Question
The core question this paper addresses is: Can pure reinforcement learning (RL) be used to enhance reasoning capabilities in LLMs without relying heavily on supervised fine-tuning (SFT)?

### Approach
The researchers took a novel approach by:
1. Applying reinforcement learning directly to base models without supervised fine-tuning as an initial step (DeepSeek-R1-Zero) (poor readability, and language
 mixing)
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

## Reward Modeling
1. Accuray Rewards
2. Format Rewards

##  AhaMomentofDeepSeek-R1-Zero
![image](https://github.com/user-attachments/assets/548175a9-d337-4fd8-9d1f-98419db0e098)

## DeepSeek-R1-Evaluation
 For all our models, the maximum generation length is set to 32,768 tokens. For benchmarks requiring sampling, we use a temperature of $0.6$, a top-p value of $0.95$, and generate 64 responses per query to estimate pass@1.
<div align="center">


| Category | Benchmark (Metric) | Claude-3.5-Sonnet-1022 | GPT-4o 0513 | DeepSeek V3 | OpenAI o1-mini | OpenAI o1-1217 | DeepSeek R1 |
|----------|-------------------|----------------------|------------|--------------|----------------|------------|--------------|
| | Architecture | - | - | MoE | - | - | MoE |
| | # Activated Params | - | - | 37B | - | - | 37B |
| | # Total Params | - | - | 671B | - | - | 671B |
| English | MMLU (Pass@1) | 88.3 | 87.2 | 88.5 | 85.2 | **91.8** | 90.8 |
| | MMLU-Redux (EM) | 88.9 | 88.0 | 89.1 | 86.7 | - | **92.9** |
| | MMLU-Pro (EM) | 78.0 | 72.6 | 75.9 | 80.3 | - | **84.0** |
| | DROP (3-shot F1) | 88.3 | 83.7 | 91.6 | 83.9 | 90.2 | **92.2** |
| | IF-Eval (Prompt Strict) | **86.5** | 84.3 | 86.1 | 84.8 | - | 83.3 |
| | GPQA-Diamond (Pass@1) | 65.0 | 49.9 | 59.1 | 60.0 | **75.7** | 71.5 |
| | SimpleQA (Correct) | 28.4 | 38.2 | 24.9 | 7.0 | **47.0** | 30.1 |
| | FRAMES (Acc.) | 72.5 | 80.5 | 73.3 | 76.9 | - | **82.5** |
| | AlpacaEval2.0 (LC-winrate) | 52.0 | 51.1 | 70.0 | 57.8 | - | **87.6** |
| | ArenaHard (GPT-4-1106) | 85.2 | 80.4 | 85.5 | 92.0 | - | **92.3** |
| Code | LiveCodeBench (Pass@1-COT) | 33.8 | 34.2 | - | 53.8 | 63.4 | **65.9** |
| | Codeforces (Percentile) | 20.3 | 23.6 | 58.7 | 93.4 | **96.6** | 96.3 |
| | Codeforces (Rating) | 717 | 759 | 1134 | 1820 | **2061** | 2029 |
| | SWE Verified (Resolved) | **50.8** | 38.8 | 42.0 | 41.6 | 48.9 | 49.2 |
| | Aider-Polyglot (Acc.) | 45.3 | 16.0 | 49.6 | 32.9 | **61.7** | 53.3 |
| Math | AIME 2024 (Pass@1) | 16.0 | 9.3 | 39.2 | 63.6 | 79.2 | **79.8** |
| | MATH-500 (Pass@1) | 78.3 | 74.6 | 90.2 | 90.0 | 96.4 | **97.3** |
| | CNMO 2024 (Pass@1) | 13.1 | 10.8 | 43.2 | 67.6 | - | **78.8** |
| Chinese | CLUEWSC (EM) | 85.4 | 87.9 | 90.9 | 89.9 | - | **92.8** |
| | C-Eval (EM) | 76.7 | 76.0 | 86.5 | 68.9 | - | **91.8** |
| | C-SimpleQA (Correct) | 55.4 | 58.7 | **68.0** | 40.3 | - | 63.7 |

</div>


### Distilled Model Evaluation


<div align="center">

| Model                                    | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 | CodeForces rating |
|------------------------------------------|------------------|-------------------|-----------------|----------------------|----------------------|-------------------|
| GPT-4o-0513                          | 9.3              | 13.4              | 74.6            | 49.9                 | 32.9                 | 759               |
| Claude-3.5-Sonnet-1022             | 16.0             | 26.7                 | 78.3            | 65.0                 | 38.9                 | 717               |
| o1-mini                              | 63.6             | 80.0              | 90.0            | 60.0                 | 53.8                 | **1820**          |
| QwQ-32B-Preview                              | 44.0             | 60.0                 | 90.6            | 54.5               | 41.9                 | 1316              |
| DeepSeek-R1-Distill-Qwen-1.5B       | 28.9             | 52.7              | 83.9            | 33.8                 | 16.9                 | 954               |
| DeepSeek-R1-Distill-Qwen-7B          | 55.5             | 83.3              | 92.8            | 49.1                 | 37.6                 | 1189              |
| DeepSeek-R1-Distill-Qwen-14B         | 69.7             | 80.0              | 93.9            | 59.1                 | 53.1                 | 1481              |
| DeepSeek-R1-Distill-Qwen-32B        | **72.6**         | 83.3              | 94.3            | 62.1                 | 57.2                 | 1691              |
| DeepSeek-R1-Distill-Llama-8B         | 50.4             | 80.0              | 89.1            | 49.0                 | 39.6                 | 1205              |
| DeepSeek-R1-Distill-Llama-70B        | 70.0             | **86.7**          | **94.5**        | **65.2**             | **57.5**             | 1633              |

</div>

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

## Official Prompts
In the official DeepSeek web/app, we don't use system prompts but design two specific prompts for file upload and web search for better user experience. In addition, the temperature in web/app is 0.6.

For file upload, please follow the template to create prompts, where {file_name}, {file_content} and {question} are arguments.

```
file_template = \
"""[file name]: {file_name}
[file content begin]
{file_content}
[file content end]
{question}"""
```

For Web Search, {search_results}, {cur_date}, and {question} are arguments.
search_answer_en_template = \
```
# The following contents are the search results related to the user's message:
{search_results}
In the search results I provide to you, each result is formatted as [webpage X begin]...[webpage X end], where X represents the numerical index of each article. Please cite the context at the end of the relevant sentence when appropriate. Use the citation format [citation:X] in the corresponding part of your answer. If a sentence is derived from multiple contexts, list all relevant citation numbers, such as [citation:3][citation:5]. Be sure not to cluster all citations at the end; instead, include them in the corresponding parts of the answer.
When responding, please keep the following points in mind:
- Today is {cur_date}.
- Not all content in the search results is closely related to the user's question. You need to evaluate and filter the search results based on the question.
- For listing-type questions (e.g., listing all flight information), try to limit the answer to 10 key points and inform the user that they can refer to the search sources for complete information. Prioritize providing the most complete and relevant items in the list. Avoid mentioning content not provided in the search results unless necessary.
- For creative tasks (e.g., writing an essay), ensure that references are cited within the body of the text, such as [citation:3][citation:5], rather than only at the end of the text. You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the search results, extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
- If the response is lengthy, structure it well and summarize it in paragraphs. If a point-by-point format is needed, try to limit it to 5 points and merge related content.
- For objective Q&A, if the answer is very brief, you may add one or two related sentences to enrich the content.
- Choose an appropriate and visually appealing format for your response based on the user's requirements and the content of the answer, ensuring strong readability.
- Your answer should synthesize information from multiple relevant webpages and avoid repeatedly citing the same webpage.
- Unless the user requests otherwise, your response should be in the same language as the user's question.

# The user's message is:
{question}
```

## Questions for the Audience
## Question 1 
What unique behavior did the DeepSeek-R1-Zero model develop that surprised researchers, as shown in the 'aha moment' example?

## Answer 1
The model developed self-correction behavior where it stopped in the middle of solving a problem, recognized it was using an overly complicated approach, and restarted with a better method - all without being explicitly programmed to do so. This emergent behavior demonstrated a form of metacognition similar to human problem-solving.

## Question 2 
What is the fundamental difference between how a model learns through accuracy rewards in reinforcement learning versus supervised fine-tuning?

## Answer 2 
In supervised fine-tuning, the model learns by imitating complete examples with correct answers provided by humans. It's like learning from a textbook with worked solutions. With accuracy rewards in reinforcement learning, the model is only given problems and feedback on whether its final answer is correct - not how to solve it. This allows the model to discover novel solution methods on its own rather than just mimicking human approaches. DeepSeek-R1-Zero demonstrated that models can develop sophisticated reasoning strategies through this exploration process without being shown examples first.

## Resources and References
- GitHub Repository: https://github.com/deepseek-ai/DeepSeek-R1
- AI@Meta. Llama 3.1 model card, 2024. URL https://github.com/meta-llama/llama-m
 odels/blob/main/models/llama3_1/MODEL_CARD.md
- Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/claude-3-5-sonnet
- M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda,
 N.Joseph, G. Brockman, A.Ray, R.Puri, G.Krueger, M.Petrov, H. Khlaaf, G. Sastry, P. Mishkin,
 B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet,
 F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss,
 A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse,
 A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage,
 M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and
 W. Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021.
 URLhttps://arxiv.org/abs/2107.03374.
- My pseudocode of Deepseek-R1 algorithms

