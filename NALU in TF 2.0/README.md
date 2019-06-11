In the `NALU_using_TF_2_0` notebook I present my implementation of [Neural Arithmetic Logic Units] (https://arxiv.org/abs/1808.00508](https://arxiv.org/abs/1808.00508) that was proposed by Trask et al. last year. The paper presents a solution to a very important problem in neural networks. Despite having the capability of approximating any arbitrary functions, neural networks show very poor performance at counting. Put in other words, they fail to extrapolate to the values that were seen by them during the training process. 

The novelty of the paper lies in two main components as proposed by the authors: Neural Accumulator (NAC) and Neural Arithmetic Logic Gates (NALU) which build on top of NAC. I would encourage the readers to take a look at the paper once, it is extremely well written and most importantly written in plain English. I would also suggest you read [this article](https://medium.com/tensorflow/understanding-neural-arithmetic-logic-units-11b0f85c1d1d) on NALU alongside the original paper. 

I used the TensorFlow 2.0 (beta) for this implementation since in TF 2.0 you get a lot of flexibility in terms customization of layers, defining forward passes and so on. 
 
I have included a link to run the notebook on Google Colab. But make sure to include the `utils.py` file when you are running it. It contains a utility function for generating toy data. 
