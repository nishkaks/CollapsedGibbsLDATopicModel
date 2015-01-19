## Latent Dirichlet Allocation and Collapsed Gibbs Sampling
The goal of this project is to implement probabilistic topic models to extract the
main themes from a large unstructured collection of documents. I have used
Latent Dirichlet Allocation(LDA) for the generative model, along with collapsed
Gibbs sampling to perform posterior inference. The algorithm would infer the
probability distribution over words associated with each topic, the distribution
over topics for each document and the topic responsible for generating each word.
I have used a subset of the brown corpus available in the NLTK toolkit as the data
set for my experiments.