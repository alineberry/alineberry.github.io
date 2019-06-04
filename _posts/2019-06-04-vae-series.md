---
title: "[DRAFT] Blog Post Series: From KL Divergence to Variational Autoencoder in PyTorch"
date: 2019-05-28
tags: [machine learning]
excerpt: "Landing page for the blog post series"
mathjax: true
author_profile: true
permalink: /vae-series
---

In this series of four posts, I attempt to build up the theory, mathematics, and intuition of variational autoencoders (VAE), starting with some basic fundamentals and then moving closer and closer to a full PyTorch implementation with each post. Without sacrificing technical rigor, I try to demystify things along the way by providing informal discussion aimed at building the reader's intuition.

<img src="{{ site.url }}{{ site.baseurl }}/images/vae/vae-architecture.png" alt="">{: .align-center}
<figcaption>Illustration of the VAE model architecture</figcaption>

The ultimate goal of the series is to provide the full picture of variational autoencoders, all the way from expected values to Python classes. The first couple of posts deal with the general theory of variational inference and deriving and understanding the evidence lower bound (ELBO). The third post transforms the general theory to VAE-specific theory. The fourth post establishes the final connections between theory and code, provides a full VAE implementation written in PyTorch, and shows some interesting experiments.

<figure class="half" style="display:flex">
  <img src="{{ site.url }}{{ site.baseurl }}/images/vae/datagen_final.png" height="100">
  <img src="{{ site.url }}{{ site.baseurl }}/images/vae/frey_face.png" height="100">
</figure>

Posts:

1. [**A Quick Primer on KL Divergence**]({{ site.url }}{{ site.baseurl }}/vae-series/kl-divergence). KL divergence is fundamental tool that is used everywhere in this series. This is a quick introduction for those who may not be familiar already.
1. [**Latent Variable Models, Expectation Maximization, and Variational Inference**]({{ site.url }}{{ site.baseurl }}/vae-series/variational-inference). This post dives into latent variable models and how to train them. It also introduces expectation maximization (EM), which is very related to the VAE.
1. [**Variational Autoencoder Theory**]({{ site.url }}{{ site.baseurl }}/vae-series/vae-theory). Here, we continue to develop the theory into the VAE objective function and discuss how the VAE model architecture is designed to achieve specific probabilistic goals.
1. [**Variational Autoencoder Code and Experiments**]({{ site.url }}{{ site.baseurl }}/vae-series/vae-theory). The culmination of the series, this post hammers home how to implement the theoretical framework in code. It also provides the code itself and shows some interesting experiments.
