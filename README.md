# Blood Cell Classification with Machine Learning

Jake Spencer Walklate

# Resources

<http://github.com/svnty/blood-cell-classification>

# Abstract

In this study, we employ a Multi-Layer Perceptron (MLP) regression model to classify cell types using the publicly available Blood Cell Detection Dataset (BCDD). The dataset comprises annotated microscopic images of various blood cell classes, providing a robust basis for supervised learning. Our approach involved preparing the data, training the MLP to recognize features in the cells, and then testing its ability to correctly classify new, unseen samples.

# Objective

Accurate identification of blood cell types is critical for medical diagnostics and treatment planning. Traditional manual classification via microscopy is labour intensive, can be subjective, and is susceptible to errors. This project investigates the application of a Multi-Layer Perceptron (MLP) model to automate blood cell classification using the Blood Cell Detection Dataset (BCDD). The goal is to assess the MLP's ability to learn distinctive features from cell images and deliver reliable classification performance, highlighting the potential of neural networks to enhance clinical decision-making with faster, more consistent outcomes. This is a method of supervised machine learning.

# Data Processing

hello

# PAC analysis

For a PAC analysis, we have a hypothesis family (), then we apply a learning algorithm which reduces us to a single hypothesis (). We then calculate if the algorithm we have arrived upon is PAC-learnable. First, we define delta as our rate of failure (frequency of not landing within the error margin), and we define epsilon as the error margin we accept.

## Samples sizes

To calculate if our algorithm is PAC-learnable, we select a sample size n.

```math
n≥(1/(2ε^2)) ln⁡(2/δ)
```

## PAC algorithm

To be PAC-learnable, the probability that our algorithm succeeds, should be frequent enough that it exceeds the defined success rate.

```math
Pr⁡[Error(k)≤ε]≥1- δ
```

# Results


# Discussions

Future iterations should split the white blood cells into their specific types, as white blood cells encompass many categories, such as granulocytes and lymphocytes. The current data set doesn't have this labelled, but future data-sets could include this.

# References



# Appendix

