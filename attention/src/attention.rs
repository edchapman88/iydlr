use anyhow::Error;
use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::{RealElement, Tensor};
use std::marker::PhantomData;

pub trait MaskedSelfAttention<T, E>: DLModule<T, E>
where
    T: Tensor<E>,
    E: RealElement,
{
}

/// Attention module, generic over the type of the elements contained within the tensors.
/// <script type="math/tex; mode=display">
/// Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
/// </script>
/// `Q,K,V dims`: (batch_size, seq_len, d_k)
pub struct MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: RealElement,
{
    pub query_weights: Vec<L>,
    pub key_weights: Vec<L>,
    pub value_weights: Vec<L>,
    pub num_heads: usize,
    pub mask: T,
    pub _marker_t: PhantomData<T>,
    pub _marker_e: PhantomData<E>,
}

impl<T, E, L> MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: RealElement,
{
    pub fn new(x: &T, num_heads: usize) -> Self {
        // Generate weights tensors W_Q, W_K, W_V with shapes (embedding_dim, d_k),
        // where d_k is embedding_dim / num_heads. For now, we assume num_heads = 1.
        // Then generate W_Q, W_K, W_V with same shape (batch x sequence x channel)
        let v = x.shape();
        let (batch_size, seq_len, embedding_dim) = (v[0], v[1], v[2]);
        let d_k = embedding_dim / num_heads;
        // TODO: no constructor currently, come back to
        // let query_weights: Vec<L> = (0..num_heads).map(|_| L::new(embedding_dim, d_k)).collect();

        todo!()
    }
}

// TODO: consider renaming as `LearnableTransform`
impl<T, E, L> DLModule<T, E> for MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: RealElement,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        // let masked_x: T = self.mask.forward(x)?;
        let masked_x: T = self.mask * x.clone(); // element-wise multiplication
        for attention_head_idx in 0..self.num_heads {
            let query = self.query_weights[attention_head_idx].forward(x).unwrap(); // just a matmul, Unwrap used since we currently do not have conversion implemented
            let key: T = self.key_weights[attention_head_idx].forward(x).unwrap();
            let value: T = self.value_weights[attention_head_idx].forward(x).unwrap();
            let last_dim_of_keys = key.shape().last().unwrap();
            // let last_dim_of_keys = key.shape().last().ok_or(anyhow!("Empty dim"))?;
            // let att: T = query.matmul(&key.transpose()) * 1 / sqrtf64(last_dim_of_keys)?; // make sure only last two dimensions are transposed
            // let att: T = query.matmul(&key.transpose()).unwrap() * 1. / E::sqrt(last_dim_of_keys);

            // Here:
            let att: T = query.matmul(&key.transpose()).unwrap() * E::from(1.)
                // TODO: make this safer
                / E::from((*last_dim_of_keys as f64).powf(-0.5));
            // matmul with V
            let att_v: T = att.matmul(&value);
        }
        // make sure only last two dimensions are transposed
        todo!()
    }

    fn params(&self) -> Vec<E> {
        todo!()
    }
}

impl<T, E, L> MaskedSelfAttention<T, E> for MultiHeadAttention<T, E, L>
where
    L: LinearLayer<T, E>,
    T: Tensor<E>,
    E: RealElement,
{
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_construct() {
        todo!()
    }

    #[test]
    fn test_forward() {
        todo!()
    }
}
