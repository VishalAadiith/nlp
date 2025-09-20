# Enhancing Detoxification Of Text Summaries With Seq2seq Language Models Using Reinforcement Learning

A research project implementing reinforcement learning-based fine-tuning to enhance the detoxification capabilities of sequence-to-sequence language models for dialogue summarization.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Research Contribution](#research-contribution)
- [Future Work](#future-work)
- [Citation](#citation)

## 🎯 Overview

This project addresses the critical issue of toxic language generation in seq2seq language models. By leveraging Reinforcement Learning (RL) techniques, specifically Proximal Policy Optimization (PPO), we fine-tune state-of-the-art models to generate less toxic summaries while preserving essential information from input dialogues.

### Problem Statement
- Seq2seq language models can inadvertently generate toxic or offensive content
- Existing post-processing techniques are computationally expensive and don't address root causes
- Need for automated detoxification that maintains content quality

### Solution
A novel RL-based fine-tuning approach that:
- Uses a RoBERTa-based hate speech classifier as reward model
- Optimizes models to generate less toxic summaries
- Preserves summarization quality while reducing harmful content

## ⭐ Key Features

- **Multi-Model Implementation**: FLAN-T5, BART, and GODEL models
- **Reinforcement Learning**: PPO-based fine-tuning with toxicity rewards
- **Automated Detoxification**: No manual intervention required
- **Quality Preservation**: Maintains ROUGE scores while reducing toxicity
- **Comprehensive Evaluation**: Both quantitative and qualitative analysis

## 📁 Project Structure

```
├── Bart/                   # BART model implementation
├── FlanT5/                # FLAN-T5 model implementation  
├── Godel_model/           # GODEL model implementation
└── README.md              # Project documentation
```

## 🤖 Models Implemented

### 1. FLAN-T5 (Fine-tuned Language Net - Text-to-Text Transfer Transformer)
- Variant of T5 model fine-tuned on diverse tasks
- **Results**: 9.40% improvement in mean toxicity score, 7.76% reduction in standard deviation

### 2. BART (Bidirectional and Auto-Regressive Transformer)  
- Denoising autoencoder for pretraining sequence-to-sequence models
- **Results**: 7.09% improvement in mean toxicity score, 26.74% reduction in standard deviation

### 3. GODEL (Generative Open-Domain Enhanced Language)
- Pre-trained language model for open-ended dialogue summarization
- **Results**: 24.68% decrease in mean toxicity score, 15.12% reduction in standard deviation (Best Performance)

## 🔬 Methodology

### 1. Data Pre-processing
- Filter dialogues within specified length range
- Wrap dialogues with summarization instructions
- Tokenize using model-specific tokenizers

### 2. Toxicity Evaluation
- Use Meta AI's RoBERTa-based hate speech model
- Toxicity scores range from 0-1 (1 = highest toxicity)
- Leverage 'nothate' class logits as positive rewards

### 3. Reinforcement Learning Fine-tuning
```
1. Generate summaries using base language model
2. Compute rewards using hate speech classifier
3. Optimize policy using PPO algorithm
4. Iterate until convergence
```

### 4. Evaluation Metrics
- **Quantitative**: Mean toxicity scores, standard deviations
- **Qualitative**: Manual analysis of summary appropriateness
- **Performance**: ROUGE scores for summarization quality

## 📊 Dataset

**DialogSum Dataset**: Benchmark for open-domain dialogue summarization
- Dialogue transcripts with corresponding summaries
- Diverse range of topics and conversation styles
- Pre-processed for optimal length and readability

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/text-detoxification-rl.git
cd text-detoxification-rl

# Install required dependencies
pip install torch transformers
pip install datasets evaluate
pip install trl  # for reinforcement learning
pip install scikit-learn pandas numpy

# For specific model requirements
pip install sentence-transformers  # for embeddings
pip install accelerate  # for model optimization
```

## 📖 Usage

### Running BART Model
```bash
cd Bart/
python train_bart.py --data_path ../data/dialogsum --epochs 10
python evaluate_bart.py --model_path ./checkpoints/bart_detoxified
```

### Running FLAN-T5 Model
```bash
cd FlanT5/
python train_flant5.py --model_name google/flan-t5-base
python evaluate_flant5.py --model_path ./models/flant5_detoxified
```

### Running GODEL Model
```bash
cd Godel_model/
python train_godel.py --base_model microsoft/GODEL-v1_1-base-seq2seq
python evaluate_godel.py --checkpoint ./checkpoints/godel_final
```

## 📈 Results

### Quantitative Improvements

| Model | Mean Toxicity Improvement | Standard Deviation Reduction |
|-------|--------------------------|------------------------------|
| FLAN-T5 | 9.40% | 7.76% |
| BART | 7.09% | 26.74% |
| **GODEL** | **24.68%** | **15.12%** |

### Qualitative Analysis
- Significant reduction in offensive language while preserving meaning
- Enhanced contextual understanding and appropriate tone
- Maintained summarization accuracy across diverse dialogue topics

## 🎓 Research Contribution

### Novel Contributions:
1. **RL-based Detoxification**: First comprehensive study applying PPO to text detoxification
2. **Multi-Model Analysis**: Comparative evaluation across three SOTA models
3. **Reward Model Design**: Effective use of hate speech classifier as reward signal
4. **Comprehensive Evaluation**: Both automatic metrics and human evaluation

### Technical Advantages:
- **Root Cause Addressing**: Tackles toxicity generation at model level
- **Computational Efficiency**: Reduces need for expensive post-processing
- **Automated Pipeline**: Minimal manual intervention required
- **Generalization**: Effective across different model architectures

## 🔮 Future Work

- [ ] **Advanced Reward Modeling**: Incorporate contextual and domain-specific knowledge
- [ ] **Multi-Domain Evaluation**: Test generalizability across diverse datasets
- [ ] **Fairness Integration**: Address bias mitigation alongside toxicity reduction
- [ ] **Real-time Deployment**: Optimize for production-ready applications
- [ ] **Multilingual Extension**: Extend approach to other languages
- [ ] **Continual Learning**: Adapt to evolving toxic language patterns



