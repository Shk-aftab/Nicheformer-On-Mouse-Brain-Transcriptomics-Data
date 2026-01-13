# Milestone 3 – Federated Fine-Tuning Framework

## Objective
Milestone 3 implements a centralized and federated fine-tuning pipeline
for the NicheFormer model using the dataset prepared in Milestone 2.

The goal is to move from data preparation to actual learning, while
maintaining clean abstractions so that Milestone 4 can extend this work
without refactoring.

## Key Outcomes
- Centralized fine-tuning baseline
- Federated fine-tuning using FedAvg
- Standardized evaluation and reporting
- Modular design for future extensions (privacy, personalization, spatial modeling)

## Dataset
- 10x Genomics Xenium Mouse Brain replicates
- Subset of SpatialCorpus-110M
- Data prepared and partitioned in Milestone 2

## Expected Usage
All components must follow the contracts defined in this folder.
No module should directly access data or models outside these contracts.
