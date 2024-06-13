use attention::attention::SelfAttention;
use config::Config;
use interfaces::{
    deep_learning::{ActivationLayer, DLModule, LinearLayer},
    tensors::{RealElement, Tensor},
};
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
struct Block<L, A, T, E, Al>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
{
    pub self_attention: A,
    pub linear_layer1: L, // i: C, o: 4C
    pub activation_layer: Al,
    pub linear_layer2: L, // i: 4C, o: C
    pub intermediate_dim: usize,
    pub num_head: usize,
    pub _marker_t: PhantomData<T>,
    _marker_e: PhantomData<E>,
}

impl<T, E, L, A, Al> DLModule<T, E> for Block<L, A, T, E, Al>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: Tensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
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
        let att: T = self.self_attention.forward(x).unwrap(); // in: (B x T x C), out: (B x T x C)
        let residual1: T = att.clone() + x.clone(); // in: (B x T x C), out: (B x T x C)

        let lin: T = self.linear_layer1.forward(&residual1).unwrap(); // in: (B x T x C), out: (B x T x 4C)
        let act: T = self.activation_layer.forward(&lin).unwrap(); // in: (B x T x 4C), out: (B x T x 4C)
        let lin2: T = self.linear_layer2.forward(&act).unwrap(); // in: (B x T x 4C), out: (B x T x C)

        let residual2: T = lin2.clone() + residual1.clone(); // in: (B x T x C), out: (B x T x C)

        Ok(residual2) // (B x T x C)
    }

    fn params(&self) -> Vec<E> {
        // pub self_attention: A,
        // pub linear_layer1: L, // i: C, o: 4C
        // pub activation_layer: Al,
        // pub linear_layer2: L, // i: 4C, o: C
        self.self_attention
            .iter()
            .flat_map(|layer| layer.params())
            .chain(self.linear_layer1.iter().flat_map(|layer| layer.params()))
            .chain(
                self.activation_layer
                    .iter()
                    .flat_map(|layer| layer.params()),
            )
            .chain(self.linear_layer2.iter().flat_map(|layer| layer.params()))
            .collect()
    }
}

// TODO: once activation is concrete
impl Block<LinLayer, MultiHeadAttention, TensorImpl, Node<f64>> {
    fn new(config: &Config, is_masked: bool) -> Self {
        let self_attention = MultiHeadAttention::new(config, is_masked);
        // Residual connection: add embedding matrix X to the output of the sub-layer element-wise
        let linear_layer1 = LinLayer::new(config.embed_dim, 4 * config.embed_dim, config.seed);
        let activation_layer = ActLayer::new();
        let linear_layer2 = LinLayer::new(4 * config.embed_dim, config.embed_dim, config.seed);
        // Residual connection: add embedding matrix X to the output of the sub-layer element-wise
    }
}

#[cfg(test)]
mod tests {
    use crate::block;

    #[test]
    fn test_construct() {
        let config = get_config();
        let attention = MultiHeadAttention::new(&config, true);
        assert_eq!(attention.num_heads, 4);
        assert!(attention.mask.is_some());
        // check that mask has the right shape
        // print the shape of the mask
        //println!("{:?}", attention.mask.as_ref().unwrap().shape());
        assert_eq!(attention.mask.unwrap().shape(), vec![7, 7]);
        assert_eq!(attention.query_weights.len(), 4);
        // println!("{:?}", attention.query_weights[0].w);
        println!("{:?}", attention.key_weights[0].w);
    }

    #[test]
    fn test_forward() {
        let config = get_config();
        let block = Block::new(&config, true);
        let x = Te::from_vec(
            &vec![config.batch_size, config.seq_len, config.embed_dim],
            &vec![Node::<f64>::zero(); config.batch_size * config.seq_len * config.embed_dim],
        );
        let out = block.forward(&x).unwrap();
        let expected_shape = vec![2, 7, 20];
        let actual_shape = out.shape();
        assert_eq!(actual_shape, expected_shape);
    }
}
