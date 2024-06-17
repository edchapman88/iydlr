use num::traits::Zero;
use std::{
    cmp::{PartialEq, PartialOrd},
    error::Error,
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, Mul, Sub},
};
use thiserror::Error;

use crate::utils::{Exp, Ln, Pow};

pub trait AsAnyhowError: From<AsStdError> + Send + Sync {}

#[derive(Error, Debug)]
#[error(transparent)]
pub struct AsStdError(#[from] anyhow::Error);
impl AsAnyhowError for AsStdError {}

/// Tensor interface, generic over the the type of the elements contained within the tensor.
/// The element type must be an implementer of `Element`.
pub trait Tensor<E>:
    Debug
    + PartialEq
    + Clone
    //+ Sized
    + IntoIterator<Item = E>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Add<E, Output = Self>
    + Sub<E, Output = Self>
    + Mul<Output = Self>
    + Mul<E, Output = Self>
    + Div<E, Output = Self>
    + Div<Output = Self>
    + Into<Vec<E>>
where
    E: Element,
{
    type TensorError: Debug + AsAnyhowError + From<AsStdError> + From<anyhow::Error>;

    fn shape(&self) -> Vec<usize>;

    fn from_vec(shape: &Vec<usize>, data: &Vec<E>) -> Result<Self, Self::TensorError>;

    // Fill a matrix by repeatedly cloning the provided element.
    // Note: the behaviour might be unexpected if the provided element clones "by reference".
    fn fill_with_clone(shape: Vec<usize>, element: E) -> Self;

    fn at(&self, idxs: Vec<usize>) -> Option<&E>;

    fn at_mut(&mut self, idxs: Vec<usize>) -> Option<&mut E>;

    fn transpose(&self) -> Self;

    fn matmul(&self, other: &Self) -> Result<Self, Self::TensorError>;

    /// Sum across one or more dimensions (eg. row-wise sum for a 2D matrix resulting in a "column
    /// vector")
    fn dim_sum(&self, dims: Vec<usize>) -> Self;

    fn concat(&self, other: &Self, dim: usize) -> Result<Self, Self::TensorError>;

    fn reshape(&mut self, new_shape: Vec<usize>);
}

/// Collection of traits required by the elements of a Tensor.
pub trait Element:
    Debug
    + Clone
    + PartialEq
    + Display
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Zero
{
}

/// A Subtrait of `Tensor`, extending the interface to include methods that require more
/// "real number like" behaviour from the tensor elements. The `RealTensor` element must be an
/// implementer of the `RealElement` trait.
pub trait RealTensor<E>: Tensor<E> + Exp + Pow<E> + Ln
where
    E: RealElement,
{
    /// Softmax across one dimension, leaving shape unchanged
    fn softmax(&self, dim: usize) -> Self;

    // Fill a tensor with calls to `MathPrimitive::from_f64`
    // Note: May provide different behaviour to `Tensor::fill_with_clone` (eg. by creating "new"
    // primitives rather than cloning existing primitives).
    // TODO(mhauru): Come back to this later.
    fn fill_from_f64(shape: Vec<usize>, data: f64) -> Self;
}

// impl<T> From<T> for Vec<f64>
// where
//     T: RealTensor,
// {
//     fn from(tensor: T) -> Self {
//         let mut vec = Vec::new();
//         for element in tensor.iter() {
//             vec.push(element.to_f64());
//         }
//         vec
//     }
// }

/// A Subtrait of `Element`, extending the trait to capture "real number like" behaviour.
pub trait RealElement: Element + Exp + Pow + Ln + From<f64> + PartialOrd {
    fn neg_inf() -> Self;
}

// Below are some implementations of `Element` and `RealElement` "for free". This should facilitate
// unit testing with these types.
impl Element for usize {}
impl Element for u32 {}
impl Element for u16 {}
impl Element for i32 {}
impl Element for f64 {}

impl RealElement for f64 {
    fn neg_inf() -> Self {
        -std::f64::INFINITY
    }
}
