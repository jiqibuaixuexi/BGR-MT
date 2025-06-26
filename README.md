# Code for "Graph-Regularized Semi-Supervised Learning with Diffusion Model Improves Multi-label Ankylosing Spondylitis Assessment on Pelvic Radiographs"

This repository contains the official implementation for our paper, which introduces a novel framework for multi-label assessment of Ankylosing Spondylitis (AS) from pelvic radiographs. Our approach leverages a graph-regularized semi-supervised learning strategy combined with a diffusion-based generative model to overcome challenges of data scarcity and annotation cost.

## Graphical Abstract

The figure below provides a comprehensive overview of our proposed methodology and key results. It illustrates the data flow, the architecture of our generative model, the overall BGR-MT learning framework, and a summary of the model's performance across various metrics.

![Graphical Abstract](images/Graphical%20Abstract.png)

## Overview

Our framework consists of two main components:

1.  **Generative Model**: A diffusion model trained to synthesize high-fidelity, realistic pelvic radiographs. This model is used to augment the training data, enabling a semi-supervised learning approach.
2.  **BGR-MT Framework**: A new semi-supervised paradigm enforcing batch-internal graph-regularized feature consistency.

## Key Features

-   **Diffusion-based Data Augmentation**: Generates realistic radiographic images to expand the training set.
-   **Semi-Supervised Learning**: Efficiently utilizes both labeled and unlabeled (generated) data.
-   **Multi-Label Assessment**: Simultaneously provides a main diagnosis and assesses four key joint regions.
-   **Graph Regularization**: Models anatomical dependencies between joints for more robust and clinically relevant predictions.

## Code and Data

We are currently in the process of cleaning and organizing the code and will release it publicly upon the paper's acceptance.

