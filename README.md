# Reversing-Convolutions-to-VAEs-

In this assignment, we will tackle the topic of auto-encoders,
a simple trick to use neural networks for unsupervised learning.
Whereas auto-encoders with fully connected layers are pretty simple,
new types of layers are necessary to build convolutional auto-encoders.
The most important of these new layers is the transposed convolution,
which is often confusingly referred to as a deconvolution.
Finally, we turn auto-encoders into generative models with some clever tricks.

## Auto-Encoders

An easy way to use supervised models in an unsupervised setting
is to invent a prediction task that only requires the inputs.
The most straightforward approach to this paradigm is to learn the identity function.
It is trivial to find a network that solves this task perfectly (linear regression).
In order to get more interesting models,
it is therefore important to make it slightly harder for the network to learn the identity.

The auto-encoder is a neural network architecture for learning these identity functions.
In general, auto-encoders consist of two parts:
the first part is the **encoder**, which maps the inputs to some *code*
and the second part is the **decoder**, which maps this code back to the inputs.
This setup is especially interesting when the code is much smaller than the input.
In this case, the code forms a bottleneck for the information flow,
and the network must learn to compress the information in the inputs to get good reconstructions.
This effectively allows to learn a lossy compression scheme where
the encoder can be used to compress the inputs and the decoder is used for decompression.

Typically, encoder and decoder will have some sort of symmetry in their architecture.
In fully-connected models, this symmetry can be obtained by transposing the weight matrices.
Convolutional layers can be transposed as well, although that might not be as obvious.
Note that we are only talking about the architecture, i.e. the shape of the weight matrices,
and not about the weights, which are typically **not shared** between encoder and decoder.

<div style="text-align: center">
  <figure style="display: inline-block; width: 49%; margin: 0">
    <img src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides.gif" />
    <figcaption style="width: 100%;"> Normal convolution </figcaption>
  </figure>
  <figure style="display: inline-block; width: 49%; margin: 0">
    <img src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides_transposed.gif" />
    <figcaption style="width: 100%; text-align: center;"> Transposed convolution </figcaption>
  </figure>
</div>

*visualisations taken from the [github](https://github.com/vdumoulin/conv_arithmetic) that comes with [this guide](https://arxiv.org/abs/1603.07285)*

------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 1: Transposed Convolutions 

Although the transpose of a matrix is typically not its inverse,
it can be interpreted as a way to *undo* a matrix multiplication.
This is especially true for fully connected layers,
where we find the tranpose of the weight matrix in the backward pass.
Using this analogy, we can find a transposed convolution
by taking the operation we find in the backward pass of a conv layer.

To get a feeling for how transposed convolutions can *undo* convolutions,
we are going to be implementing a method from Zeiler and Fergus (from ZF net).
They used transposed convolutions to visualise activations in the input space.
However, we will also need to undo max-pooling layers and ReLU non-linearities.
Therefore, we first focus on how to undo different kinds of layers.

------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 2: Visualising Features 

In the previous exercise I mentioned Zeiler and Fergus.
They used this kind of undoable modules to visualise features.
I think it is worth trying this out for yourself.
Is there anything we can learn from these visualisations?

------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 3: Convolutional Auto-Encoders 

Architecturally an auto-encoder is not much more than a model
where inputs and outputs have the same dimensions.
For fully-connected networks this is relatively straightforward.
However, with our newly acquired knowledge on transposed convolutions,
also convolutional and pooling layers should not pose too much problems.
One key difference is that the transposed convolutions n an auto-encoder
will have their own learnable parameters.

The architecture is typically symmetrical and consists of two parts:

 1. an **encoder** that transforms the image to some latent space and
 2. a **decoder** that produces an image from vectors in the latent space.

Since both components can operate independently from each other,
auto-encoders can be used to e.g. learn compression algorithms.
Because the output of both networks can be useful,
it is typically a good idea to have them predict logits,
i.e. not to use activation functions at the end.
This way, both the output of the encoder
and the output of the decoder can be used in loss functions.

------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 4: Auto-Encoding 

Of course, training an auto-encoder takes more than just a nice architecture.
The loss function as well as the pre-processing of data will also affect the results.
There is no right or wrong as long as you get the desired result.
However, we can make certain assumptions to guide the search for a good combination.
After all, all components have to work together to get nice reconstructions.
Can you spot/explain the difference in results for different assumptions?

------------------------------------------------------------------------------------------------------------------------------------------------------------
## Variational Auto-Encoders

Variational auto-encoders (VAEs) extend auto-encoders in a probabilistic way.
Although this might sound complicated, it only requires a few modifications to make an auto-encoder variational.
First of all, the latent space is regularised to stay close to a (standard normal) distribution.
Secondly, VAEs do not produce specific codes, but rather a distribution of codes.
In practice this is done by directly mapping inputs to the parameters of a distribution.

The main advantage of this approach, is that the the distribution of the latent codes remains well under control.
As a result, we can sample from this latent distribution and decode these sampled codes to generate new data.

### Exercise 5: Variational Auto-Encoder 

VAEs predict the distribution parameters of the latent space,
rather than a specific code in the latent space.
In the case of a Gaussian latent distribution,
this means that the encoder produces a mean and log-variance for every input sample.
By predicting the log-variance, the variance is guaranteed to be positive.

Because the codes of a VAE are distributions, it is not possible to directly decode them back to images.
Instead, specific codes have to be sampled from the latent distribution before decoding.
This is the main difference, architecturally, between AEs and VAEs.
The main difference during training is the regularisation.
This is what makes VAEs especially interesting for generating new images.

------------------------------------------------------------------------------------------------------------------------------------------------------------
### Exercise 6: Balance 

In order to incorporate the regularisation loss in our trainer,
I provided a (hacky) loss function wrapper: `RegularisedLoss`.
This wrapper makes it possible to observe the original loss
together with the regularisation loss.
This will be useful because regularisation is always a balancing act.
Therefore, the goal of this final exercise is to train the VAE
in such a way that reconstruction and regularisation losses are balanced.









