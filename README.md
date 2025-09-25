# Blood Cell Classification with Machine Learning

Jake Spencer Walklate

# Resources

<http://github.com/svnty/blood-cell-classification>

<https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset>

# Abstract

In this study, we employ a Multi-Layer Perceptron (MLP) regression model to classify cell types using the publicly available Blood Cell Detection Dataset (BCDD). The dataset comprises annotated microscopic images of various blood cell classes, providing a robust basis for supervised learning. Our approach involved preparing the data, training the MLP to recognize features in the cells, and then testing its ability to correctly classify new, unseen samples.

# Introduction

Accurate identification of blood cell types is critical for medical diagnostics and treatment planning. Traditional manual classification via microscopy is labour intensive, can be subjective, and is susceptible to errors. This project investigates the application of a Multi-Layer Perceptron (MLP) model to automate blood cell classification using the Blood Cell Detection Dataset (BCDD). The goal is to assess the MLP's ability to learn distinctive features from cell images and deliver reliable classification performance, highlighting the potential of neural networks to enhance clinical decision-making with faster, more consistent outcomes. 

# Method

In this project, we use a Multi-Layer Perceptron (MLP) model for blood cell classification. An MLP is a type of fully connected neural network, where each neuron in one layer is connected to every neuron in the next layer. The network consists of an input layer, one or more hidden layers, and an output layer.

Each neuron receives inputs from the previous layer, multiplies each input by a learned weight, sums the results, adds a bias, and passes the result through a nonlinear activation function (such as ReLU). This nonlinearity allows the network to learn complex relationships in the data. The output of each neuron becomes the input for the next layer, allowing information to flow forward through the network.

During training, the network uses backpropagation to adjust its weights. The output of the network is compared to the true label, and the error is propagated backward through the layers. The weights are updated to minimize this error, allowing the network to learn from the data and improve its predictions over time.

In the final output layer, the network produces a probability distribution over the possible cell classes using the softmax function. The class with the highest probability is selected as the prediction.

## Data

### Data source

The dataset used is the [Blood Cell Detection Dataset (BCDD)](https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset), downloaded from Kaggle. It contains 100 high-resolution microscopic images of blood cells, each annotated with bounding boxes for individual blood cells. The annotations specify the cell type (e.g., red blood cell, white blood cell) and the coordinates of each cell in the image. The dataset provides a total of 2340 annotated cell regions.

### Data Processing

The data processing pipeline involved several key steps:

1. **Dataset Extraction:** The dataset was downloaded and extracted using the Kaggle API. 
2. **Annotation Analysis:** The `annotations.csv` file was loaded with pandas to inspect the distribution of cell types. Each annotation provides the image filename, cell type label, and bounding box coordinates.
3. **Image Cropping:** For each annotated cell, the corresponding region was cropped from the original image using OpenCV, resized to 64x64 pixels, and stored as a training sample. This produced a dataset of cell images suitable for input to a neural network.
4. **Normalization:** All images were activated to have pixel values in the [0, 1] range. Labels were encoded as integers and then one-hot encoded for classification.

### Data splitting

To prevent data leakage, the dataset was split at the image level: 80% of images were used for training and 20% for testing. This ensures that no cell from a given image appears in both the training and test sets. The split was performed using a custom function that shuffles image filenames and assigns them to train/test groups, then filters the annotations accordingly.

## Model

### Architecture

We implemented a Multi-Layer Perceptron (MLP) using TensorFlow and Keras. The model architecture consists of:

- Input layer that flattens the image data
- Two fully connected (dense) layers with ReLU activation and dropout for regularization
- A final softmax output layer for multi-class classification

The model was compiled with the Adam optimizer and categorical cross-entropy loss. Early stopping and learning rate reduction callbacks were used to prevent overfitting and optimize training.

### Neurons

In a neuron layer of a Multi-Layer Perceptron (MLP), each neuron receives inputs from all neurons in the previous layer. For the input layer, each pixel value from the image is treated as an input, that is each neuron receives every pixel as input. Each neuron computes a weighted sum of its inputs, adds a bias, and then applies a nonlinear activation function (such as ReLU):

```math
z = \sum_{i=1}^{N} w_i x_i + b
```

### ReLU vs sigmoid vs tanh

ReLU activation was used in all hidden layers for its efficiency and ability to mitigate vanishing gradients. The output layer uses softmax for multi-class classification on the logits. ReLU is defined as the max of the pre-activation linear output, any values less than 0 become 0.

```math
ReLU: a = f(z) = max(0, z)
```

### Backpropagation

Backpropagation is the core algorithm (A) used to train neural networks, including Multi-Layer Perceptrons (MLPs). After the model computes its output and the loss is calculated, backpropagation works by differentiating the loss with respect to each weight in the network, moving backwards from the output layer to the input layer.

This process uses the chain rule from calculus to efficiently compute gradients for each neuron in every layer. The chain rule allows the algorithm to propagate the error signal through the network, layer by layer, so that each neuron's weights are updated in proportion to their contribution to the final error. This enables the network to learn complex mappings from input to output by iteratively adjusting its parameters to minimize the loss.

Backpropagation is essential for deep learning, as it makes it possible to train large networks with many layers and millions of parameters.

### Final layer

In the final layer of the MLP, the network produces a probability distribution over all possible cell classes using the softmax function. The softmax function takes the raw outputs (logits) from the last dense layer and transforms them into probabilities that sum to 1. This allows the model to interpret its output as the likelihood of each class.

Mathematically, the softmax function is defined as:

```math
p_i = softmax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
```

The class with the highest probability is selected as the model's prediction. Softmax is essential for multi-class classification tasks, as it enables the network to output interpretable probabilities for each class.

### Loss function

For this multi-class classification task, we use the categorical cross-entropy loss function (L). This loss function measures the difference between the predicted probability distribution `p_i` (output by the softmax layer) and the true class labels (one-hot encoded). Minimizing categorical cross-entropy encourages the model to assign high probability to the correct class and low probability to incorrect classes.

Mathematically, for a single sample with true class `y_i` and predicted probabilities `p_i`, the loss is:

```math
L = -\sum_{i=1}^{C} y_i \log(p_i)
```

where `C` is the number of classes, `y_i` is 1 if class `i` is the true class and 0 otherwise (due to one-hot encoding), and `p_i` is the predicted probability for class `i`.

This loss is differentiable and works well with backpropagation, making it the standard choice for neural networks in multi-class classification problems.

### Training monitoring

During training, we use the EarlyStopping callback to monitor the validation loss (`val_loss`). After each epoch, the model checks if the validation loss has improved. If it does not improve for a set number of epochs (defined by the patience parameter), training is stopped early to prevent overfitting. This ensures the model does not continue training once it stops making progress on unseen validation data, leading to better generalization and more efficient training.

## PAC analysis

For a PAC analysis, we have a hypothesis family (H) which we defined as a MLP, then we apply our learning algorithm (A) which reduces us to a single hypothesis (h). We then calculate if the algorithm we have arrived upon is PAC-learnable. First, we define delta as our rate of failure (frequency of not landing within the error margin), and we define epsilon as the error margin we accept.

### Samples sizes

To calculate if our algorithm is PAC-learnable, we select a sample size n.

```math
n≥(1/(2ε^2)) ln⁡(2/δ)
```

### Algorithm

To be PAC-learnable, the probability that our algorithm succeeds, should be frequent enough that it exceeds the defined success rate.

```math
Pr⁡[Error(k)≤ε]≥1-δ
```

### Results

PAC-Learnability Analysis:

- Chosen epsilon (error margin): 0.05
- Chosen delta (failure probability): 0.05
- Required sample size for PAC-learnability: n >= 738
- Actual training samples used: 1882

The training set size meets the PAC-learnability requirement for these parameters.

# Results

The MLP achieved high accuracy on the test set, with strong performance across all cell classes. The training process was monitored using accuracy and loss curves, and a confusion matrix was generated to visualize classification performance. The model demonstrated reliable generalization to unseen cell images, confirming the effectiveness of the approach.

Our final function has a validation accuracy of 99.56%, with a validation loss of 2.65% at epoch 48.

# Discussions

Future iterations should split the white blood cells into their specific types, as white blood cells encompass many categories, such as granulocytes and lymphocytes. The current data set doesn't have this labelled, but future data-sets could include this.

Limitations include the relatively small dataset size, the lack of fine-grained WBC subtypes and the highly controlled image positions and lack of variety. The current pipeline is robust and reproducible, and can be extended to more complex classification tasks as more data becomes available.

As our images are 256x256 pixels and have 3 colour channels (R, G, B), our first layer has 196,608 input neurons. The MLP then scales exponentially as the layers increase, it is possible that our model contains as many as 100 million parameters. Our network of neurons currently at 6,423,810 params could be considered enormous.

# Conclusion

This project demonstrates that a MLP can accurately classify blood cell types from microscopic images using only bounding box annotations and image crops. The approach is fully automated, reproducible, and achieves high accuracy, supporting the use of deep learning for medical image analysis. With further data and more granular labels, this method could be extended to more detailed hematological diagnostics.

# References

1. Draaslan, M. (2020). Blood Cell Detection Dataset (BCDD). Kaggle. https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset
2. Chollet, F. (2015). Keras. https://keras.io
3. Abadi, M., et al. (2016). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. https://tensorflow.org
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
5. OpenCV Library. https://opencv.org

# Appendix

See the notebook (`notebook.ipynb`) for full code, data exploration, model training, and evaluation details.

