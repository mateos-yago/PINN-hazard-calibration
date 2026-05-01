# CLAUDE.md - Project Guidelines

## 1. Code Organization & Architecture

The project should be developed in an object-oriented approach, with logic stored in different modules and libraries in a logical way that allows for easier maintenance, extension, and manipulation of the code. Everything should have clean documentation, but should avoid documentation that is unnecessarily long—keep everything short and efficient.

## 2. Neural Networks & Performance

Applications involving neural networks should be developed using PyTorch. Apart from that, all code should attempt to be as efficient as possible (although avoiding making it too cryptic). Operations should be vectorized or tensorized when possible.

## 3. Results Management & Experimentation

Results should always be exported, and the export file structure should be sound, logical, and well thought through. Every experiment should be documented and logged, informing about the hyperparameters that it used. The trained neural network weights for each experiment should be saved so they can be accessed easily at any point. Consider saving some comments about the rationale behind each experiment.

## 4. Version Control

Every change to the code should be pushed to the GitHub remote repo with clean explanatory commits. Try to do these commits often, so it is easy to come back to previous versions of the code if needed.
