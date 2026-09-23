# TensorFlow From Zero

**Absolutely. Let's understand TensorFlow from zero, without assuming you already know deep learning.**

---

## 🧠 What is TensorFlow?

TensorFlow is a software library/framework used to build and train machine learning and deep learning models.

In simple words:

> TensorFlow helps us create neural networks, give them data, train them, and use them to make predictions.

For example, suppose you want a computer to recognize whether an image contains a cat 🐱 or dog 🐶.

You could use TensorFlow to:

```
Images
   ↓
TensorFlow
   ↓
Neural Network
   ↓
Training
   ↓
Learn patterns
   ↓
New Image
   ↓
Cat / Dog
```

---

## 1. Why do we need TensorFlow?

You can build a very simple machine learning algorithm using Python and NumPy yourself.

For example:

```python
prediction = x * weight + bias
```

But neural networks become much more complicated.

A neural network might contain:

```
Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
Output
```

Each layer can contain thousands or millions of numbers called **weights**.

During training, we repeatedly need to:

1. Perform mathematical calculations  
2. Calculate the error  
3. Calculate gradients  
4. Update weights  
5. Repeat this many times  

Doing all of this manually would be extremely difficult.  
**TensorFlow provides tools that handle these operations for us.**

---

## 2. Why is it called "TensorFlow"?

The name has two parts:

### Tensor

A **tensor** is basically a container for numbers.  
You can think of tensors as generalized arrays.

**Examples:**

| Type                  | Example                          | Dimension     |
|-----------------------|----------------------------------|---------------|
| One number            | `5`                              | 0-dimensional |
| List                  | `[1, 2, 3]`                      | 1-dimensional |
| Table                 | `[[1, 2, 3], [4, 5, 6]]`         | 2-dimensional |
| Image                 | `224 × 224 × 3` (RGB)            | 3-dimensional |

---

## 3. What does "Flow" mean?

**Flow** refers to data flowing through mathematical operations.

For example:

```
Input
  ↓
Multiply
  ↓
Add
  ↓
Activation
  ↓
Output
```

So you can roughly think of TensorFlow as:

> Numbers/tensors flowing through mathematical operations.

---

## 4. TensorFlow and Neural Networks

This is where TensorFlow becomes particularly useful.

Suppose we want to create a neural network:

```
Input
  ↓
Dense Layer
  ↓
ReLU
  ↓
Dense Layer
  ↓
Sigmoid
  ↓
Output
```

TensorFlow allows us to write this relatively easily:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

Don't worry if this looks confusing right now. Let's break it down.

---

## 5. What is `tf`?

When we write:

```python
import tensorflow as tf
```

we are importing TensorFlow.  
`tf` is simply a short name for TensorFlow.

Instead of writing:

```python
tensorflow.keras
tensorflow.constant
tensorflow.data
```

we write:

```python
tf.keras
tf.constant
tf.data
```

---

## 6. What is Keras?

You will frequently see:

```python
tf.keras
```

**Keras** is TensorFlow's high-level API for building neural networks.

Think of it like this:

```
TensorFlow
│
├── Tensors
├── Mathematical operations
├── Automatic differentiation
├── GPU support
├── Data pipelines
└── Keras
     │
     └── Easy neural network building
```

Keras makes neural network development much easier.  
For beginners, you'll probably use Keras inside TensorFlow most of the time.

---

## 7. What is a layer?

A neural network is made up of **layers**.

For example:

```
Input
 ↓
Dense Layer
 ↓
Dense Layer
 ↓
Output
```

A layer performs mathematical operations on the input.

```python
tf.keras.layers.Dense(10)
```

means:  
**Create a fully connected neural network layer with 10 neurons.**

---

## 8. What is a neuron?

A neuron performs something roughly like:

```
Input
  ↓
Weighted sum
  ↓
Activation function
  ↓
Output
```

Mathematically:

```
z = w₁x₁ + w₂x₂ + w₃x₃ + b
```

Then an activation function is applied:

```
output = activation(z)
```

TensorFlow handles these calculations for us.

---

## 9. What happens during training?

This is the most important part.

Suppose we're training a model to recognize cats and dogs.

We give it:

```
Image → Dog
```

Initially, the model might predict:

```
Cat: 0.70
Dog: 0.30
```

That's wrong.

TensorFlow calculates how wrong the prediction was using a **loss function**.

```
Prediction
     ↓
Loss function
     ↓
Error
```

Then TensorFlow calculates **gradients**.  
These gradients tell the model how its weights should change.

```
Loss
 ↓
Gradients
 ↓
Update weights
 ↓
Better prediction
```

This happens again and again:

```
Data
 ↓
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Weight update
 ↓
Prediction
 ↓
Loss
 ↓
...
```

Eventually, the model hopefully learns useful patterns.

---

## 10. What is automatic differentiation?

This is one of TensorFlow's important features.

During neural network training, we need derivatives/gradients.

For example:

```
Loss
  ↓
How should weight W1 change?
How should weight W2 change?
How should weight W3 change?
...
```

For a large neural network, there could be millions of weights.  
TensorFlow can **automatically** calculate these gradients.

Example:

```python
with tf.GradientTape() as tape:
    prediction = model(x)
    loss = loss_function(y, prediction)

gradients = tape.gradient(loss, model.trainable_variables)
```

You don't need to manually calculate every derivative.

---

## 11. What are tensors actually used for?

Almost everything in a neural network is represented using tensors.

```
Tensor
 ↓
Neural network
 ↓
Tensor
 ↓
Prediction
```

Example:

```python
x = tf.constant([[1, 2, 3]])
print(x)
```

You might see:

```
tf.Tensor([[1 2 3]], shape=(1, 3), dtype=int32)
```

Here:

```
shape = (1, 3)
```

means:

```
1 row
3 values
```

---

## 12. TensorFlow can work with GPUs

Deep learning requires a lot of mathematical calculations.

A CPU can do them, but GPUs can perform many suitable calculations in parallel.

```
CPU
 ↓
Some calculations

GPU
 ↓↓↓↓↓↓↓↓↓
Many calculations simultaneously
```

TensorFlow can use available hardware acceleration for supported workloads.

That's particularly useful for:

- CNNs  
- RNNs  
- Transformers  
- Large neural networks  
- Computer vision  
- Deep learning  

---

## 13. What can you build with TensorFlow?

### 🖼️ Computer Vision

```
Image
 ↓
CNN
 ↓
Cat/Dog
```

**Applications:**
- Image classification  
- Object detection  
- Image segmentation  
- Face-related applications  

### 📝 Natural Language Processing

```
"I love this movie"
        ↓
Neural Network
        ↓
Positive
```

**Applications:**
- Sentiment analysis  
- Text classification  
- Language modeling  
- Text generation  
- Sequence processing  

### 🔢 Time-series prediction

```
Previous stock/weather/sensor values
              ↓
          Neural Network
              ↓
       Future prediction
```

---

## 14. TensorFlow vs Scikit-learn

Since you're learning ML, this distinction is important.

### Scikit-learn

Mostly used for traditional machine learning algorithms:

- Linear Regression  
- Logistic Regression  
- KNN  
- SVM  
- Decision Tree  
- Random Forest  
- Naive Bayes  

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

### TensorFlow

Primarily useful for neural networks and deep learning:

- Neural Networks  
- CNN  
- RNN  
- LSTM  
- Deep learning models  

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

**Simple mental model:**

```
Scikit-learn
      ↓
Traditional ML

TensorFlow/Keras
      ↓
Neural Networks / Deep Learning
```

There is some overlap, but this is a useful beginner mental model.

---

## 15. A complete TensorFlow workflow

When building a neural network, you'll commonly follow this pattern:

```
       DATA
        ↓
  Preprocessing
        ↓
     Dataset
        ↓
  Build Model
        ↓
     Compile
        ↓
      Train
        ↓
    Evaluate
        ↓
    Predict
```

In code:

```python
import tensorflow as tf

# 1. Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 2. Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 3. Train
model.fit(X_train, y_train, epochs=10)

# 4. Evaluate
model.evaluate(X_test, y_test)

# 5. Predict
model.predict(X_test)
```

Don't worry about understanding every line yet. You'll learn them one by one.

---

## 16. What is `compile()`?

Before training, we need to tell the model **how it should learn**.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

There are three important things here:

### Optimizer

```python
optimizer="adam"
```

The optimizer controls how the model updates its weights.

> "How should I change my weights to reduce the error?"

### Loss

```python
loss="binary_crossentropy"
```

Loss tells us:

> "How wrong is the model?"

### Metrics

```python
metrics=["accuracy"]
```

Metrics tell us how well the model is performing.

---

## 17. What is `fit()`?

This is where training happens:

```python
model.fit(X_train, y_train, epochs=10)
```

Think of `fit()` as:

> "Learn from this training data."

`epochs=10` means the model goes through the training dataset **10 times**.

---

## 18. What is `predict()`?

After training:

```python
model.predict(X_test)
```

means:

> "Use what you learned to make predictions."

```
Input
 ↓
Trained model
 ↓
Prediction
```

---

## 19. What is a Dataset?

TensorFlow also provides tools for handling data efficiently.

You'll often see:

```python
tf.data.Dataset
```

Example:

```python
dataset = tf.data.Dataset.from_tensor_slices(
    ([1, 2, 3, 4], [0, 1, 0, 1])
)
```

Then you can do things like:

```python
dataset = dataset.shuffle(4)
dataset = dataset.batch(2)
```

This is where the `buffer_size` you asked about earlier comes into the picture.

```python
dataset.shuffle(buffer_size=1000)
```

means TensorFlow uses a buffer of up to 1000 elements when shuffling.

---

## 20. TensorFlow in one picture

Think of TensorFlow like a deep-learning toolbox:

```
                 TENSORFLOW
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    Tensors        Keras        tf.data
       │             │             │
       ↓             ↓             ↓
   Numbers       Neural         Data
   & arrays      Networks       Pipeline
                     │
                     ↓
              ┌──────────────┐
              │    Training  │
              └──────────────┘
                     │
             ┌───────┴───────┐
             ↓               ↓
          Gradients       Optimizer
             │               │
             └───────┬───────┘
                     ↓
              Updated Weights
                     ↓
              Better Model
```

---

## 🧠 The easiest way to remember TensorFlow

Remember these **5 things**:

1. **Tensor** → A container for numbers.  
2. **Neural network** → A collection of layers that learns patterns.  
3. **Loss** → Measures how wrong the model is.  
4. **Gradient** → Tells the model how its weights should change.  
5. **Optimizer** → Updates the weights to reduce the loss.  

So the learning process becomes:

```
Input
  ↓
Neural Network
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Optimizer
  ↓
Updated Weights
  ↓
Better Neural Network
```

That's the core idea behind TensorFlow-based deep learning.

---

Since you're currently learning RNNs, Word2Vec/CBOW and deep learning, the next useful step would be to understand:

**TensorFlow tensors → layers → model → compile → loss → optimizer → fit()**

with one very small example, before jumping into CNN/RNN code.