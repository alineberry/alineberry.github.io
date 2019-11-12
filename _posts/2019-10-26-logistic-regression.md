---
title: "Logistic Regression Deep Dive"
date: 2019-10-26
tags: [machine learning]
excerpt: "..."
mathjax: true
author_profile: false
---

Logistic regression is possibly the most well-known machine learning model for classification tasks. The classic case is binary classification, but it can easily be extended to multiclass or even multilabel classification settings. It's quite popular in the data science, machine learning, and statistics community for many reasons:

- The mathematics are relatively simple to understand
- It is quite interpretable, both globally and locally; it is not considered a "black box" algorithm.
- Training and inference are very fast
- It has a minimal number of model parameters, which limits its ability to overfit

In many cases, logistic regression can perform quite well on a task. However, more complex architectures can typically perform better if tuned properly. Thus, practioners and researchers alike commonly use logistic regression as a baseline model.

Logistic regression's architecture is a basic form of a neural network and it is optimized the same way as a neural network. This makes it a valuable case study for those interested in deep learning.

In this post I will be deriving logistic regression's objective function from Maximum Likelihood Estimation (MLE) principles, deriving the gradient of the objective with respect to the model parameters, and visualizing how a gradient descent update shifts the decision boundary for a misclassified point.

## Objective Function

Consider a dataset of $$n$$ data points $$ \{ (x_i, y_i) \}_{i=1}^n $$ where  $$ x_i \in \mathbb{R}^d $$ are the observed features and $$ y_i \in \{0, 1\} $$ are the corresponding binary labels.

Logistic regression is a discriminative classifier (as opposed to a generative one), meaning that it models the posterior $$p(y \lvert x)$$ directly. It models this posterior as a linear function of $$x$$ "squashed" into $$ [0, 1] $$ by the sigmoid function (denoted as $$\sigma$$).

The model is parameterized by a vector $$ w \in \mathbb{R}^{d+1} $$. For notational convenience we will assume each $$x_i$$ is concatenated with a $$1$$ so that the model's bias term is contained in the final component of the vector $$w_{d+1}$$.

$$
p(y=1 \lvert x) = \sigma(w^\top x) \\
\sigma(w^\top x) = \frac{1}{1 + \exp(-w^\top x)} \\
$$

Using the notational convenience that $$ y \in \{0, 1\} $$, we can write the model equation more generally:

$$
p(y \lvert x) = \sigma(w^\top x)^y(1-\sigma(w^\top x))^{1-y}
$$

Notice the equivalence with the [Bernoulli PMF](https://en.wikipedia.org/wiki/Bernoulli_distribution#Properties). This means logistic regression is modeling the data with a Bernoulli likelihood. Assuming data are iid, the likelihood of the full dataset is given by:

$$
p(y_1, \dots, y_n \lvert x_1, \dots, x_n) =
\prod_{i=1}^n \sigma(w^\top x_i)^{y_i}(1-\sigma(w^\top x_i))^{1-y_i} \\
$$

There is no closed form solution to logistic regression, so the log likelihood is optimized instead via gradient descent. The log likelihood objective function $$J$$ is given by:

$$
\begin{align}

J &= \log p(y_1, \dots, y_n \lvert x_1, \dots, x_n) \\
&= \log \prod_{i=1}^n \sigma(w^\top x_i)^{y_i}(1-\sigma(w^\top x_i))^{1-y_i} \\
&= \sum_{i=1}^n \log \sigma(w^\top x_i)^{y_i}(1-\sigma(w^\top x_i))^{1-y_i} \\
&= \sum_{i=1}^n \log \sigma(w^\top x_i)^{y_i} + \log (1-\sigma(w^\top x_i))^{1-y_i} \\
&= \sum_{i=1}^n y_i \log \sigma(w^\top x_i) + (1-y_i) \log (1-\sigma(w^\top x_i)) \\

\end{align}
$$

## Gradient of the Objective Function

In this section, we'll be deriving the gradient of the objective function with respect to the model parameters $$w$$. The derivation makes liberal use of the chain rule and other basic multivariate calculus rules.

The gradient of a sum is equal to the sum of the gradients, so for simplicity let's consider a single data point.

Simplify notation:

$$
\sigma = \sigma(w^\top x)
$$

The derivative of the sigmoid function will come in handy:

$$
\frac{d}{dz} \sigma (z) = \sigma(z) (1 - \sigma(z))
$$

Derivation:

$$
\begin{align*}

\nabla_w J &= \nabla_w y \log \sigma + \nabla_w (1-y) \log (1-\sigma) \\
&= \frac{y}{\sigma}\nabla_w\sigma + \frac{1-y}{1-\sigma}\nabla_w(1-\sigma) \\
&= \frac{y}{\sigma}\nabla_w\sigma - \frac{1-y}{1-\sigma}\nabla_w\sigma \\
&= (\frac{y}{\sigma} - \frac{1-y}{1-\sigma})\nabla_w\sigma \\
&= \frac{y(1-\sigma) - (1-y)\sigma}{\sigma(1-\sigma)} \nabla_w\sigma \\
&= \frac{y - \sigma y - \sigma + \sigma y}{\sigma(1-\sigma)} \nabla_w\sigma \\
&= \frac{y - \sigma}{\sigma(1-\sigma)}\nabla_w\sigma \\
&= \frac{y - \sigma}{\sigma(1-\sigma)} \sigma(1-\sigma) x \\
&= (y - \sigma) x

\end{align*}
$$

## Visualizing a Gradient Update
