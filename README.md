# Policy Optimization Lab

This repository contains Jupyter notebook implementations and explanations of reinforcement learning, policy gradient methods, preference optimization, and LLM post-training algorithms.

The goal of this project is to build a clear understanding of reinforcement learning methods from first principles, starting from fundamentals such as rewards, returns, advantages, policy gradients, and importance sampling, and then progressing toward modern policy optimization algorithms used for language model alignment and reasoning.

The notebooks cover both classical reinforcement learning algorithms and recent LLM post-training methods, including REINFORCE, PPO, DPO, GRPO, Dr. GRPO, DAPO, CISPO, OPD, TIS, ICE-POP, and OPSD.

This repository is designed as a learning and research lab. Each notebook focuses on one algorithm or concept, with an emphasis on mathematical intuition, objective functions, tensor-level implementation details, and the connection between the algorithm and practical LLM training.

## Topics Covered

* Reinforcement learning fundamentals
* Rewards, returns, and advantages
* Policy gradients
* REINFORCE
* Importance sampling
* Proximal Policy Optimization
* Direct Preference Optimization
* Group Relative Policy Optimization
* Dr. GRPO
* DAPO
* CISPO
* OPD
* Truncated importance sampling
* ICE-POP
* OPSD
* LLM post-training objectives
* Reward modeling and preference optimization
* KL penalties and clipped objectives
* Advantage normalization
* On-policy and off-policy optimization

## Purpose

Modern LLM post-training is increasingly built around reinforcement learning and policy optimization. Algorithms such as PPO, GRPO, Dr. GRPO, and DAPO build on core RL ideas but adapt them for language models, preference data, verifiable rewards, and long-form reasoning.

This repository breaks those methods into focused notebooks so each algorithm can be understood independently and compared against related approaches.

The focus is on clarity, derivation, and implementation rather than production-scale distributed training.

## Intended Audience

This repository is useful for:

* Learning reinforcement learning from first principles
* Understanding policy gradient algorithms
* Studying RLHF and LLM post-training methods
* Preparing for ML research and engineering interviews
* Comparing PPO, DPO, GRPO, Dr. GRPO, DAPO, and related algorithms
* Building intuition for modern reasoning-model training

## Repository Style

The repository is notebook-first. Each notebook is intended to be readable, self-contained, and focused on a specific concept or algorithm.

Where possible, notebooks include:

* Mathematical objective
* Intuition behind the algorithm
* Step-by-step PyTorch implementation
* Tensor shape walkthrough
* Small runnable examples
* Notes on stability and practical training behavior
