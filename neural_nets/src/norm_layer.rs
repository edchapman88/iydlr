// This module contains the activation layer struct and its implementation.
use interfaces::{
    deep_learning::{DLModule, LinearLayer},
    tensors::{Element, RealElement, RealTensor, Tensor},
};
use std::iter::Iterator;
use std::marker::PhantomData;

use crate::lin_layer::LinLayer;

pub struct NormLayer<T: Tensor<E>, E: Element> {
    tensor_phantom: PhantomData<T>,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for NormLayer<T, E>
where
    T: RealTensor<E>,
    E: RealElement + Into<f64>,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    // fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
    //     let shape = x.shape();
    //     let sum = x.dim_sum(vec![shape.len() - 1]);
    //     //let mean = x.clone() / sum;
    //     let mean = x.clone() / sum;
    //     let diff: T = x.clone() - mean.clone();
    //     let diff_squared = diff.clone() * diff.clone();
    //     let sd = (diff_squared.clone()
    //         / (diff_squared.dim_sum(vec![shape.len() - 1]) + E::from(f64::EPSILON)))
    //     .pow(E::from(0.5));

    //     Ok(diff / sd)
    // }

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let t_small = E::from(f64::EPSILON);
        let n_dims = x.shape().len();
        let sum = x.dim_sum(vec![n_dims - 1]);
        let mean = x.clone() / (sum + t_small.clone());
        let diff: T = x.clone() - mean.clone();
        let diff_squared = diff.clone() * diff.clone();
        let diff_squared_sum = diff_squared.dim_sum(vec![n_dims - 1]);
        let sd = (diff_squared.clone() / (diff_squared_sum + t_small.clone()) + t_small.clone())
            .pow(E::from(0.5));

        Ok(diff / sd)
    }

    // The activation layer has no parameters, return an empty vector
    fn params(&self) -> Vec<E> {
        Vec::new()
    }
}

impl<T, E> LinearLayer<T, E> for NormLayer<T, E>
where
    T: RealTensor<E>,
    E: RealElement + Into<f64>,
{
}

impl<T, E> NormLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    pub fn new() -> Self {
        Self {
            tensor_phantom: PhantomData,
            tensor_element_phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensors::TensorImpl;

    // #[test]
    // fn construct_act_layer() {
    //     // The activation layer is constructed without any errors
    //     let layer: ActLayer<TensorImpl<f64>, f64> = ActLayer::new();
    // }

    // #[test]
    // fn three_dim_forward() {
    //     // The forward method of the activation layer is called without any errors
    //     let layer: ActLayer<TensorImpl<f64>, f64> = ActLayer::new();
    //     let x = TensorImpl::from_vec(&vec![2, 2, 2], &vec![6.0; 8]).unwrap();
    //     println!("{:?}", x.shape());
    //     let out = layer.forward(&x).unwrap();
    //     assert_eq!(out.shape(), vec![2, 2, 2]);
    // }

    // #[test]
    // fn no_negative_values() {
    //     // The ReLU function is applied to the input tensor
    //     let layer: ActLayer<TensorImpl<f64>, f64> = ActLayer::new();
    //     let x = TensorImpl::from_vec(&vec![1, 2, 2], &vec![-2.0, -1.0, 0.0, 1.0]).unwrap();
    //     let out = layer.forward(&x).unwrap();
    //     assert_eq!(
    //         out,
    //         TensorImpl::from_vec(&vec![1, 2, 2], &vec![0.0, 0.0, 0.0, 1.0]).unwrap()
    //     )
    // }
}
