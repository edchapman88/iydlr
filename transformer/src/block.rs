use attention::attention::{El, La, Mal, Te};
use attention::attention::{MultiHeadAttention, SelfAttention};
use config::Config;
use interfaces::{
    deep_learning::{ActivationLayer, DLModule, LinearLayer},
    tensors::{RealElement, Tensor},
};

use neural_nets::norm_layer::{self, NormLayer};
use neural_nets::{act_layer::ActLayer, lin_layer::LinLayer};
use std::marker::PhantomData;

// keras_nlp.layers.TransformerEncoder(
//     intermediate_dim,
//     num_heads,
//     dropout=0,
//     activation="relu",
//     layer_norm_epsilon=1e-05,
//     kernel_initializer="glorot_uniform",
//     bias_initializer="zeros",
//     normalize_first=False,
//     **kwargs
// )

/// Expecting input as a Tensor with shape (B x T x C) where:
///   - B is batch size
///   - T is time
///   - C is channel.
/// Example from the keras API ([encoder](https://keras.io/api/keras_nlp/modeling_layers/transformer_encoder/))
pub struct Block<L, A, T, E, Al, N>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
    N: LinearLayer<T, E>,
{
    pub self_attention: A,
    pub linear_layer1: L, // i: C, o: 4C
    pub activation_layer: Al,
    pub linear_layer2: L, // i: 4C, o: C
    pub norm_layer: N,
    pub intermediate_dim: usize,
    pub num_head: usize,
    pub _marker_t: PhantomData<T>,
    _marker_e: PhantomData<E>,
    _marker_n: PhantomData<N>,
}

impl<T, E, L, A, Al, N> DLModule<T, E> for Block<L, A, T, E, Al, N>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
    N: LinearLayer<T, E>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // A block consists of a self-attention layer followed by a feed-forward neural network.
        // It also implements residual connections after each sub-layer.
        // The residual connection adds the original embedding matrix x to the output of the sub-layer.
        // The feed forward neural network consists of two linear layers with a ReLU activation in between.
        // The first linear layer expands to 4 times the embedding dimension,
        // and the second linear layer projects back to the original embedding dimension.

        // TODO: implement residual connections
        println!("{}", "-".repeat(10));
        let att: T = self.self_attention.forward(x).unwrap(); // in: (B x T x C), out: (B x T x C)
        let residual1: T = att.clone() + x.clone(); // in: (B x T x C), out: (B x T x C)
        let normed_res1: T = self.norm_layer.forward(&residual1).unwrap();
        println!("{}", "*".repeat(10));
        let lin: T = self.linear_layer1.forward(&normed_res1).unwrap(); // in: (B x T x C), out: (B x T x 4C)
        let act: T = self.activation_layer.forward(&lin).unwrap(); // in: (B x T x 4C), out: (B x T x 4C)
        let lin2: T = self.linear_layer2.forward(&act).unwrap(); // in: (B x T x 4C), out: (B x T x C)
        let residual2: T = lin2.clone() + residual1.clone(); // in: (B x T x C), out: (B x T x C)
        let normed_res2: T = self.norm_layer.forward(&residual2).unwrap();
        println!("{}", "-".repeat(10));
        Ok(normed_res2) // (B x T x C)
    }

    fn params(&self) -> Vec<E> {
        // pub self_attention: A,
        // pub linear_layer1: L, // i: C, o: 4C
        // pub activation_layer: Al,
        // pub linear_layer2: L, // i: 4C, o: C
        self.self_attention
            .params()
            .into_iter()
            .chain(self.linear_layer1.params().into_iter())
            .chain(self.activation_layer.params().into_iter())
            .chain(self.linear_layer2.params().into_iter())
            .collect()
    }
}

// TODO: once activation is concrete
// Block<L, A, T, E, Al>
impl Block<La, Mal, Te, El, ActLayer<Te, El>, NormLayer<Te, El>> {
    pub fn new(config: &Config, is_masked: bool) -> Self {
        let self_attention = MultiHeadAttention::new(config, is_masked);
        // Residual connection: add embedding matrix X to the output of the sub-layer element-wise
        let linear_layer1 = LinLayer::new(config.embed_dim, 4 * config.embed_dim, config.seed);
        let activation_layer = ActLayer::new();
        let linear_layer2 = LinLayer::new(4 * config.embed_dim, config.embed_dim, config.seed);
        let norm_layer = NormLayer::new();
        // Residual connection: add embedding matrix X to the output of the sub-layer element-wise
        Self {
            self_attention,
            linear_layer1,
            activation_layer,
            linear_layer2,
            intermediate_dim: config.embed_dim * 4,
            norm_layer,
            num_head: config.num_head,
            _marker_t: PhantomData,
            _marker_e: PhantomData,
            _marker_n: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use autodiff::node::Node;
    use num_traits::Zero;

    use super::*;

    fn get_config() -> Config {
        Config {
            batch_size: 2,
            vocab_size: 10,
            seq_len: 7,
            embed_dim: 20,
            num_head: 4,
            seed: 0,
            num_blocks: 1,
        }
    }

    #[test]
    fn test_construct() {
        let config = get_config();
        // query + values + keys + lin layer 1 + lin layer 2
        // 7 * 7
        let block = Block::new(&config, true);
        println!("{}", block.params().len());
    }

    #[test]
    fn test_forward() {
        let config = get_config();
        let block = Block::new(&config, true);
        let x = Te::from_vec(
            &vec![config.batch_size, config.seq_len, config.embed_dim],
            &vec![Node::<f64>::zero(); config.batch_size * config.seq_len * config.embed_dim],
        )
        .unwrap();
        let out = block.forward(&x).unwrap();
        let expected_shape = vec![2, 7, 20];
        let actual_shape = out.shape();
        assert_eq!(actual_shape, expected_shape);
    }
}
