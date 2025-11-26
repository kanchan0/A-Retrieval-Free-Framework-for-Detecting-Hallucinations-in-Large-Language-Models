# A-Retrieval-Free-Framework-for-Detecting-Hallucinations-in-Large-Language-Models

**Author**  
**Kanchan Kumar Tiwari**  
Department of Computer Science and Engineering  
Indian Institute of Technology Madras  

Email:  
- cs23s025@smail.iitm.ac.in  
- kanchantiwari@cse.iitm.ac.in  

**Date:** November 2025  

---

## Introduction

In recent years, large language models (LLMs) have moved far beyond being research curiosities and are now powerful tools used in search engines, programming assistants, scientific writing, and everyday decision support. Their fluency creates an illusion of reliability, and many users, especially non-experts, tend to treat their responses as authoritative. Unfortunately, even the strongest LLMs occasionally produce statements that sound correct but are factually unsupported. This behavior, widely known as *hallucination*, has emerged as one of the most important obstacles to deploying LLMs in sensitive or high-precision environments.

During my initial reading on this topic, I noticed that most existing detection systems depend heavily on external sources of truth: web search APIs, curated knowledge bases, citation mining, or manual expert evaluation. These approaches work, but they introduce several problems: scalability becomes an issue, external resources might not cover specialized domains, and real-time systems cannot always afford the overhead of retrieval. More importantly, the presence of an external truth source fundamentally limits generalization across new domains or offline environments.

In an attempt to solve the above in this assignment, I am presenting a detailed framework I call **Kanchan's-HALL-Detect**, which relies purely on intrinsic and self-referential signals from the model. I intentionally avoid retrieval methods completely. The entire design is grounded in the observable behavior of LLMs, aiming for a practical tool that can be deployed even in scenarios where external fact sources are unavailable.

---

## Problem Statement

The goal of this work is to design a retrieval-free framework for detecting hallucinations using only the intrinsic behavioural signals of the model itself. Given a prompt $P$ and an LLM-generated response $R$, the task is to extract factual claims $c_i$ from $R$ and assign each claim a hallucination likelihood score $H(c_i) \in [0,1]$.


The detection must rely solely on internal indicators such as:

- token-level uncertainty  
- generation consistency across samples  
- self-verification prompts  

and should **not** consult any external fact sources.

Formally, we aim to construct a function:

$$
H(c_i) = f(c_i; P, R, M)
$$

where $f(\cdot)$ operates entirely within the model $M$'s own outputs and intermediate signals. The objective is to build a lightweight, domain-agnostic hallucination detector suitable for environments where external verification is impractical or unavailable.


---

## Proposed Framework – Kanchan's-HALL-Detect

To address the challenge of hallucination detection without relying on external truth sources, I propose a retrieval-free framework named **Kanchan's-HALL-Detect** (*Hybrid, All-Signals, Lightweight Hallucination Detector*). The core philosophy of this framework is that an LLM's own behaviour contains rich signals that can be exploited to judge the factual stability of its outputs. Instead of matching responses against external databases, the framework examines intrinsic uncertainty patterns, consistency across sampled generations, and the model's own self-evaluation.

Given a prompt $P$ and a generated response $R$, the framework extracts factual claims $c_i$ and computes a hallucination score $H(c_i)$ for each claim. This score is a combination of multiple intrinsic indicators derived directly from the model. The design is modular, allowing each component to contribute complementary evidence toward the final prediction.


### Framework Overview

The framework consists of the following sequential modules:

1. **Claim Extraction** – Identify atomic factual statements within the response $R$.
2. **Token-Level Uncertainty Analysis** – Measure log-probabilities, entropy, and logit gaps associated with the tokens of each claim.  
3. **Sampling-Based Self-Consistency** – Generate multiple alternative responses to assess whether the model remains consistent in repeating the same claims.  
4. **Self-Verification** – Query the model to explicitly rate and justify its confidence in each claim.  
5. **Score Aggregation** – Combine all intrinsic signals into a final hallucination likelihood using a weighted logistic function.

### Design Philosophy

Kanchan's-HALL-Detect is based on the observation that hallucinations often arise from internal instability rather than intentional fabrication. By analysing how the model behaves at the token level, how it varies across repeated generations, and how it evaluates its own statements, we can detect hallucination tendencies without any external fact-checking mechanisms.

### Processing Pipeline

The overall flow of Kanchan's-HALL-Detect is illustrated below:

![pipeline](pipeline.pdf)
Each module operates independently and contributes a distinct aspect of the model's internal behaviour. Together, they create a comprehensive signal ensemble that reflects the likelihood of factual instability.

## Module Independence and Complementarity

A key strength of this framework is that no single module is expected to capture all hallucination phenomena. For example:

- **Token-level statistics** highlight local uncertainty.
- **Sampling consistency** detects global instability.
- **Self-verification** exposes contradictions in the model's own reasoning.
- **Representational classifiers** reveal latent patterns encoded in the hidden layers.

By combining these signals, Kanchan's-HALL-Detect avoids overreliance on any one method and produces a robust hallucination score that generalises across prompts, tasks, and domains.

## Formal Goal

For each extracted claim $c_i$, the framework seeks to estimate:

$$
H(c_i) = f_{\theta}(c_i; P, R, M)
$$

where $f_{\theta}$ is a composite function aggregating uncertainty metrics, consistency scores, self-assessed confidence, and representation-based features.  
The output $H(c_i) \in [0,1]$ reflects the likelihood that the claim is factually unstable, as inferred solely from the model's intrinsic behaviour.

This retrieval-free approach ensures that **Kanchan's-HALL-Detect** remains lightweight, domain-independent, and suitable for deployment in privacy-restricted or offline environments.
.

---

# Claim Extraction

The hallucination detection process begins by isolating factual claims from the model-generated response \(R\). Since hallucinations typically arise at the statement level, each claim must be examined independently.

## Procedure

**1. Sentence Splitting**  
The response is first divided into sentences using simple punctuation-based rules. This avoids heavyweight NLP dependencies and keeps the system lightweight.

**2. Factual Sentence Identification**  
A sentence is treated as a factual claim if it includes indicators such as:  
(i) named entities (persons, places, organisations),  
(ii) numerical or date expressions,  
(iii) relational verbs like “is”, “was”, “founded”, “invented”.

**3. Atomic Claim Formation**  
If a sentence contains multiple assertions, it is split into separate minimal claims.

Example:  
`"kanchan is IITian who plays cricket for IIT Madras."` becomes two claims:
**c₁:** Kanchan is an IITian  
**c₂:** Kanchan plays cricket for IIT Madras

## Output

The output of this module is a clean list of atomic factual claims:

`C = {c₁, c₂, ..., c_N}`

which are then evaluated individually by downstream modules.

# Token-Level Uncertainty Analysis

This module evaluates how confident the model was while generating each claim. Since hallucinations often correlate with unstable token predictions, we analyse token-level statistics directly derived from the model's logits.

## Intuition

Claims generated with low log-probabilities, high entropy, or small logit gaps tend to be less stable. These metrics form the first intrinsic layer of hallucination detection and provide a token-level confidence profile for each claim.

---

# Sampling-Based Self-Consistency

Hallucinations often manifest as instability across repeated generations. If the model cannot consistently reproduce the same factual claim when prompted multiple times, the claim is likely unreliable. This module evaluates the stability of each extracted claim through sampling-based consistency checks.

## Procedure

**1. Multiple Generations**  
For the same prompt $P$, the model is sampled $k$ times to obtain alternative responses:

$$
R^{(1)},\ R^{(2)},\ \dots,\ R^{(k)}
$$

**2. Claim Alignment**  
From each sampled response, the corresponding factual statement related to claim $c$ is extracted. These constitute comparison claims:

$$
c^{(1)},\ c^{(2)},\ \dots,\ c^{(k)}
$$
## Interpretation

A low agreement score indicates that the model frequently changes its answer across samples, suggesting that the claim lacks internal stability. This consistency check forms an important complementary signal to token-level uncertainty.

---

# Self-Verification

In addition to uncertainty and consistency signals, the model's own self-assessment can provide useful information about potential hallucinations. By prompting the model to explicitly rate its confidence in a claim, we obtain an introspective signal that often highlights uncertainty not visible in the original response.

## Procedure
For each extracted claim $c$, the model is queried with:

Claim: <c>
On a scale from 0 to 1, how confident are you that this claim is correct?
Provide only the number on the first line and give 1–2 sentences explaining your reasoning.
The model's numeric output is parsed as a confidence value:

$$
conf\_self(c) \in [0,1]
$$
## Interpretation

A low self-reported confidence or heavily hedged justification suggests that the claim may be unstable or weakly grounded. This self-verification step complements earlier modules by capturing the model's own awareness of its uncertainty.

---

# Score Aggregation

After collecting signals from token-level uncertainty, sampling-based consistency, and self-verification, the final step is to combine these indicators into a single hallucination likelihood score for each claim.

## Feature Vector
For every claim $c$, a compact feature vector is constructed using the outputs of the previous modules:

$$
X(c) =
\begin{bmatrix}
f_1(c) \\
f_2(c) \\
f_3(c) \\
f_4(c) \\
f_5(c)
\end{bmatrix}
$$

where:

- `f₁ = 1 − norm(mean_logp)`
- `f₂ = norm(entropy)`
- `f₃ = 1 − norm(logit_gap)`
- `f₄ = 1 − agreement`
- `f₅ = 1 − conf_self`

Each component is scaled to the range `[0,1]` for consistency. Higher values indicate a greater likelihood of instability.
## Final Scoring Function

A weighted logistic model computes the hallucination score:

$$
H(c) = \sigma\left(W^\top X(c) + b\right)
$$

where $\sigma(\cdot)$ is the sigmoid function.

## Interpretation

- **$H(c) \approx 1$** → strong indicators of instability  
- **$H(c) \approx 0$** → claim appears internally stable  

This aggregated score is the core output of **Kanchan's-HALL-Detect**, enabling retrieval-free hallucination detection.

---

# Conclusion

This report introduced **Kanchan's-HALL-Detect**, a retrieval-free framework for identifying hallucinations in large language models. Instead of relying on external knowledge sources, the system evaluates signals generated by the model itself—uncertainty, sampling consistency, and self-verification—to estimate trustworthiness.

Its modular structure, lightweight design, and domain independence make it suitable for offline, privacy-sensitive, and specialised environments. As a whole, Kanchan's-HALL-Detect offers an efficient and self-contained method for identifying potentially unreliable claims.

---

# References

1. **Manakul, P., Liusie, A., & Gales, M. J. F.**  
   *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*.  
   arXiv:2303.08896, 2023.  
   https://arxiv.org/abs/2303.08896

2. **Orgad, H., Toker, M., Gekhman, Z., Reichart, R., Szpektor, I., Kotek, H., & Belinkov, Y.**  
   *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations*.  
   arXiv:2410.02707, 2025.  
   https://arxiv.org/abs/2410.02707

3. **Liang, Y., Song, Z., Wang, H., & Zhang, J.**  
   *Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for Hallucination Mitigation*.  
   arXiv:2401.15449, 2024.  
   https://arxiv.org/abs/2401.15449

