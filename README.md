# NovoGrad for Keras

Keras port of [NovoGrad](https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/optimizers/novograd.py), from the paper [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks.](https://arxiv.org/abs/1905.11286)

# NovoGrad

<img src="https://github.com/titu1994/keras_novograd/blob/master/images/novograd.png?raw=true" height=100% width=100%>

The above image is from the paper. NovoGrad makes the optimizer more resilient to choice of initial learning rate as it behaves similarly to SGD but with gradient normalization per layer. It extends ND-Adam and also decouples weight decay from regularization. Also, it has only half the memory cost as compared to Adam, and similar memory requirements to SGD with Momentum. This allows larger models to be trained without compromizing training efficiency.

## Usage

Add the `novograd.py` script to your project, and import it. Can be a dropin replacement for `Adam` Optimizer. 

Note that NovoGrad also supports "AMSGrad"-like behaviour with the `amsgrad=True` flag.

```python
from novograd import NovoGrad

optm = NovoGrad(lr=1e-2)
```


# Requirements
- Keras 2.2.4+ & Tensorflow 1.14+ (Only supports TF backend for now).
- Numpy
