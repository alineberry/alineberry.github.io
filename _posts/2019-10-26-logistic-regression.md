---
title: "Logistic Regression Deep Dive"
date: 2019-10-26
tags: [machine learning]
excerpt: "..."
mathjax: true
author_profile: true
---

$$
y \in \{0, 1\} \\

p(y=1 \lvert x) = \sigma(w^Tx) \\

p(y \lvert x) = \sigma(w^Tx)^y(1-\sigma(w^Tx))^{1-y} \\

p(y_1, \dots, y_n \lvert x_1, \dots, x_n) =
\prod_{i=1}^n \sigma(w^Tx_i)^{y_i}(1-\sigma(w^Tx_i))^{1-y_i} \\

\begin{align}

J &= \log p(y_1, \dots, y_n \lvert x_1, \dots, x_n) \\
&= \log \prod_{i=1}^n \sigma(w^Tx_i)^{y_i}(1-\sigma(w^Tx_i))^{1-y_i} \\
&= \sum_{i=1}^n \log \sigma(w^Tx_i)^{y_i}(1-\sigma(w^Tx_i))^{1-y_i} \\
&= \sum_{i=1}^n \log \sigma(w^Tx_i)^{y_i} + \log (1-\sigma(w^Tx_i))^{1-y_i} \\
&= \sum_{i=1}^n y_i \log \sigma(w^Tx_i) + (1-y_i) \log (1-\sigma(w^Tx_i)) \\

\end{align}

$$

Gradient of sum is sum of gradients, so consider a single data point. Simplify notation:

$$
\sigma = \sigma(w^Tx)
$$

$$
\begin{align*}

\nabla_w J &= \nabla_w y \log \sigma + \nabla_w (1-y) \log (1-\sigma) \\
&= \frac{y}{\sigma}\nabla_w\sigma + \frac{1-y}{1-\sigma}\nabla_w(1-\sigma) \\
&= \frac{y}{\sigma}\nabla_w\sigma - \frac{1-y}{1-\sigma}\nabla_w\sigma \\
&= (\frac{y}{\sigma} - \frac{1-y}{1-\sigma})\nabla_w\sigma \\
&= \frac{y(1-\sigma) - (1-y)\sigma}{\sigma(1-\sigma)} \nabla_w\sigma \\
&= \frac{y - \sigma y - \sigma + \sigma y}{\sigma(1-\sigma)} \nabla_w\sigma \\
&= \frac{y - \sigma}{\sigma(1-\sigma)}\nabla_w\sigma \\
&= \frac{y - \sigma}{\sigma(1-\sigma)} \sigma(1-\sigma) \\
&= y - \sigma

\end{align*}
$$
