use crate::block::Block;
use attention::attention::SelfAttention;
use attention::attention::{El, La, Mal, Te};
use config::Config;
use embeddings::pos_encoding::PELayer;
use interfaces::deep_learning::{ActivationLayer, DLModule};
use interfaces::deep_learning::{EmbeddingLayer, LinearLayer};
use interfaces::tensors::{RealElement, RealTensor, Tensor};
use neural_nets::embedding_table::EmbeddingTable;
use neural_nets::norm_layer::NormLayer;
use neural_nets::{act_layer::ActLayer, lin_layer::LinLayer, serial::Serial};
use std::default::Default;
use std::marker::PhantomData;

pub struct Transformer<L, A, T, E, Al, N>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    // Note RealTensor used here so that softmax can be called
    T: RealTensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
    N: LinearLayer<T, E>,
{
    model: Serial<T, E>,
    _marker_l: std::marker::PhantomData<L>,
    _marker_a: std::marker::PhantomData<A>,
    _marker_al: std::marker::PhantomData<Al>,
    _marker_n: std::marker::PhantomData<N>,
}

impl Transformer<La, Mal, Te, El, ActLayer<Te, El>, NormLayer<Te, El>> {
    pub fn new(config: &Config) -> Self {
        let mut modules: Vec<
            Box<dyn DLModule<Te, El, DLModuleError = <Te as Tensor<El>>::TensorError>>,
        > = vec![];
        modules.push(Box::new(EmbeddingTable::new(
            config.embed_dim,
            config.vocab_size,
            config.seed,
        )));
        modules.push(Box::new(PELayer::<Te, El>::new()));

        for i in 0..config.num_blocks {
            modules.push(Box::new(Block::new(config, i == 0)));
        }
        modules.push(Box::new(LinLayer::new(
            config.embed_dim,
            config.vocab_size,
            config.seed,
        )));
        let model = Serial::new(modules);
        Self {
            model,
            _marker_a: PhantomData,
            _marker_al: PhantomData,
            _marker_l: PhantomData,
            _marker_n: PhantomData,
        }
    }
}

impl<T, E, L, A, Al, N> DLModule<T, E> for Transformer<L, A, T, E, Al, N>
where
    L: LinearLayer<T, E>,
    A: SelfAttention<T, E>,
    T: RealTensor<E>,
    E: RealElement,
    Al: ActivationLayer<T, E>,
    N: LinearLayer<T, E>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        Ok(self.model.forward(x)?.softmax(2))
    }

    fn params(&self) -> Vec<E> {
        self.model.params()
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
            seq_len: 8,
            embed_dim: 8,
            vocab_size: 12,
            num_head: 4,
            num_blocks: 1,
            seed: 0,
        }
    }

    #[test]
    fn test_construct() {
        let config = get_config();
        let model = Transformer::new(&config);
        println!("{}", model.params().len());
    }

    #[test]
    fn test_forward() {
        let config = get_config();
        let model = Transformer::new(&config);
        let x = Te::from_vec(
            &vec![config.batch_size, config.seq_len, 1],
            &vec![Node::<f64>::zero(); config.batch_size * config.seq_len * 1],
        )
        .unwrap();
        let out = model.forward(&x).unwrap();
        let out_vec: Vec<_> = out.clone().into();
        println!(
            "{:?}",
            out_vec
                .into_iter()
                .map(|node| node.val())
                .collect::<Vec<_>>()
        );
        let expected_shape = vec![2, 8, 12];
        let actual_shape = out.shape();
        assert_eq!(actual_shape, expected_shape);
    }
}
