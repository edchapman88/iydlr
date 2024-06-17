use anyhow::Error;
use interfaces::tensors::{AsStdError, Element, RealElement, RealTensor, Tensor};
use interfaces::utils::{Exp, Ln, Pow};
use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
    vec::Vec,
};

/// Implementation of multidimensional arrays as row major strided vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorImpl<E>
where
    E: Element,
{
    shape: Vec<usize>,
    data: Vec<E>,
}

fn strides_from_shape(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1];
    for i in 0..(shape.len() - 1) {
        strides.push(strides[i] * shape[shape.len() - 1 - i]);
    }
    return strides.into_iter().rev().collect();
}

fn num_elements_from_shape(shape: &Vec<usize>) -> usize {
    return shape.iter().product::<usize>();
}

impl<E: Element> TensorImpl<E> {
    fn strides(&self) -> Vec<usize> {
        return strides_from_shape(&self.shape);
    }

    fn num_dims(&self) -> usize {
        return self.shape.len();
    }

    fn num_elements(&self) -> usize {
        return self.data.len();
    }

    fn vec_indx(&self, idxs: Vec<usize>) -> usize {
        let s = self.strides();
        return idxs
            .iter()
            .zip(s.iter())
            .fold(0, |acc, (idx, stride)| acc + idx * stride);
    }

    /// Multiply the tensor by the transpose of a matrix.
    ///
    /// This is equivalent to `self.matmul(other.transpose())`, but faster.
    fn matmul_transpose(
        &self,
        other: &Self,
    ) -> Result<TensorImpl<E>, <TensorImpl<E> as Tensor<E>>::TensorError> {
        let self_num_dims = self.num_dims();
        let other_num_dims = other.num_dims();
        if self_num_dims < 2 {
            return Err(
                Error::msg("The first tensor int matmul must have at least 2 dimensions").into(),
            );
        }
        if other_num_dims != 2 {
            return Err(Error::msg("The second tensor int matmul must have 2 dimensions").into());
        }
        let dim1 = self.shape[self_num_dims - 2];
        let dim_inner = self.shape[self_num_dims - 1];
        let dim2 = other.shape[other_num_dims - 2];
        if dim_inner != other.shape[other_num_dims - 1] {
            return Err(Error::msg("The contracted dimensions of the tensors must match.").into());
        }
        let lead_dim = self
            .shape
            .iter()
            .take(self_num_dims - 2)
            .into_iter()
            .product::<usize>();

        // Create an unallocated data vector for the result.
        let n_elements = lead_dim * dim1 * dim2;
        let mut new_data: Vec<E> = Vec::with_capacity(n_elements);
        let mut new_shape: Vec<usize> =
            self.shape.iter().take(self_num_dims - 2).cloned().collect();
        new_shape.push(dim1);
        new_shape.push(dim2);

        // Loop over the elements in the order of new_data.
        let self_strides = self.strides();
        let other_strides = other.strides();
        let lead_stride = if self_num_dims > 2 {
            self_strides[self_num_dims - 3]
        } else {
            1
        };
        for i in 0..lead_dim {
            for j1 in 0..dim1 {
                for j2 in 0..dim2 {
                    let mut accumulator = E::zero();
                    for j_inner in 0..dim_inner {
                        let self_idx = i * lead_stride
                            + j1 * self_strides[self_num_dims - 2]
                            + j_inner * self_strides[self_num_dims - 1];
                        let other_idx = j2 * other_strides[other_num_dims - 2]
                            + j_inner * other_strides[other_num_dims - 1];

                        // TODO: since AddAssign is not impl for Node currently, just use Add.
                        // Revert this once AddAssign is implemented.
                        // accumulator += self.data[self_idx].clone() * other.data[other_idx].clone();
                        accumulator = accumulator
                            + self.data[self_idx].clone() * other.data[other_idx].clone();
                    }
                    new_data.push(accumulator);
                }
            }
        }
        return Result::Ok(TensorImpl {
            shape: new_shape,
            data: new_data,
        });
    }


    pub fn elementwise_binary_op(self, other: Self, op: fn(E, E) -> E) -> Self {
        if self.shape() == other.shape() {
            return self.elementwise_binary_op_same_shape(other, op);
        } else {
            return self.elementwise_binary_op_broadcast(other, op);
        }
    }

    fn elementwise_binary_op_same_shape(self, other: Self, op: fn(E, E) -> E) -> Self {
        let data: Vec<E> = self
            .data
            .iter()
            .zip(other.data.iter())
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|(a, b)| op(a.clone(), b.clone()))
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }

    fn elementwise_binary_op_broadcast(self, other: Self, op: fn(E, E) -> E) -> Self {
        if self.num_dims() != other.num_dims() {
            panic!("Shapes are not compatible for element-wise operations.");
        }
        let num_dims = self.num_dims();
        for i in 0..self.shape.len() {
            if self.shape[i] != other.shape[i] && self.shape[i] != 1 && other.shape[i] != 1 {
                panic!("Shapes are not compatible for element-wise operations.");
            }
        }

        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(other.shape.iter())
            .map(|(a, b)| std::cmp::max(a, b).clone())
            .collect();
        let result_num_elements = num_elements_from_shape(&new_shape);
        let mut new_data: Vec<E> = Vec::with_capacity(result_num_elements);

        let mut self_idx: Vec<usize> = vec![0; num_dims];
        let mut other_idx: Vec<usize> = vec![0; num_dims];
        let mut which_index_to_increment: usize;
        while new_data.len() != result_num_elements {
            let self_element = self.at(self_idx.clone()).unwrap();
            let other_element = other.at(other_idx.clone()).unwrap();
            new_data.push(op(self_element.clone(), other_element.clone()));

            // Increment the indices
            which_index_to_increment = num_dims - 1;
            while self_idx[which_index_to_increment] == self.shape[which_index_to_increment] - 1
                && other_idx[which_index_to_increment] == other.shape[which_index_to_increment] - 1
            {
                if which_index_to_increment == 0 {
                    break;
                } else {
                    which_index_to_increment -= 1;
                }
            }
            if self.shape[which_index_to_increment] != 1 {
                self_idx[which_index_to_increment] += 1;
            }
            for i in (which_index_to_increment + 1)..num_dims {
                self_idx[i] = 0;
            }
            if other.shape[which_index_to_increment] != 1 {
                other_idx[which_index_to_increment] += 1;
            }
            for i in (which_index_to_increment + 1)..num_dims {
                other_idx[i] = 0;
            }
        }
        let result = Self {
            shape: new_shape,
            data: new_data,
        };
        return result;
    }
}

impl<E: Element> IntoIterator for TensorImpl<E> {
    type Item = E;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> std::vec::IntoIter<Self::Item> {
        self.data.into_iter()
    }
}

impl<E: Element> From<TensorImpl<E>> for Vec<E> {
    fn from(value: TensorImpl<E>) -> Self {
        value.data.into_iter().collect()
    }
}

/// Adding to two tensors together.
impl<E: Element> Add for TensorImpl<E> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.elementwise_binary_op(other, |a, b| a + b)
    }
}

/// Dividing to two tensors elementwise.
impl<E: Element> Div for TensorImpl<E> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.elementwise_binary_op(other, |a, b| a / b)
    }
}

/// Multiplying to two tensors elementwise.
impl<E: Element> Mul for TensorImpl<E> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.elementwise_binary_op(other, |a, b| a * b)
    }
}

/// Subtracting to two tensors elementwise.
impl<E: Element + Sub<Output = E>> Sub for TensorImpl<E> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.elementwise_binary_op(other, |a, b| a - b)
    }
}

/// Adding to a scalar to a tensors together.
impl<E: Element> Add<E> for TensorImpl<E> {
    type Output = Self;

    fn add(self, scalar: E) -> Self {
        let data: Vec<E> = self
            .data
            .iter()
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|a| a.clone() + scalar.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

/// Adding to a scalar to a tensors together.
impl<E: Element> Sub<E> for TensorImpl<E> {
    type Output = Self;

    fn sub(self, scalar: E) -> Self {
        let data: Vec<E> = self
            .data
            .iter()
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|a| a.clone() - scalar.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

/// Multiplying tensor by a scalar.
impl<E: Element> Mul<E> for TensorImpl<E> {
    type Output = Self;

    fn mul(self, scalar: E) -> Self {
        let data: Vec<E> = self
            .data
            .iter()
            // TODO(mhauru) What's the consequence of cloning here? Does it affect performance?
            .map(|a| a.clone() * scalar.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

/// Dividing tensor by a scalar.
impl<E: Element> Div<E> for TensorImpl<E> {
    type Output = Self;

    fn div(self, scalar: E) -> Self {
        if scalar == E::zero() {
            panic!("Division by zero.");
        }
        let data: Vec<E> = self
            .data
            .iter()
            .map(|a| a.clone() / scalar.clone())
            .collect();
        // TODO: Remove the unwrap, and return a Result instead
        TensorImpl::from_vec(&self.shape(), &data).unwrap()
    }
}

impl<E> Tensor<E> for TensorImpl<E>
where
    E: Element,
{
    type TensorError = AsStdError;

    fn from_vec(shape: &Vec<usize>, data: &Vec<E>) -> Result<Self, Self::TensorError> {
        if num_elements_from_shape(shape) != data.len() {
            return Err(Error::msg(
                "The length of the `data` param does not match the values of the `shape` param",
            )
            .into());
        } else {
            Ok(TensorImpl {
                shape: shape.clone(),
                data: data.clone(),
            })
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Fill a matrix by repeatedly cloning the provided element.
    /// Note: the behaviour might be unexpected if the provided element clones "by reference".
    fn fill_with_clone(shape: Vec<usize>, element: E) -> Self {
        let data = vec![element; num_elements_from_shape(&shape)];
        TensorImpl { shape, data }
    }

    fn at(&self, idxs: Vec<usize>) -> Option<&E> {
        return self.data.get(self.vec_indx(idxs));
    }

    fn at_mut(&mut self, idxs: Vec<usize>) -> Option<&mut E> {
        let index = self.vec_indx(idxs);
        return self.data.get_mut(index);
    }

    fn transpose(&self) -> Self {
        if self.shape.len() < 2 {
            return self.clone();
        }
        let mut new_shape = self.shape.clone();
        let num_dims = self.num_dims();
        // Swap the last two elements of the shape vector
        new_shape.swap(num_dims - 1, num_dims - 2);
        // Compute the product of all the dimensions _not_ being transposed.
        let lead_dim = new_shape
            .iter()
            .take(num_dims - 2)
            .into_iter()
            .product::<usize>();

        // Create an unallocated data vector the same size as the original.
        let n_elements = self.num_elements();
        let mut new_data: Vec<E> = Vec::with_capacity(n_elements);

        // Loop over the elements in the order of new_data.
        let strides = self.strides();
        let lead_stride = if num_dims > 2 {
            strides[num_dims - 3]
        } else {
            1
        };
        for i in 0..lead_dim {
            for j in 0..new_shape[num_dims - 2] {
                for k in 0..new_shape[num_dims - 1] {
                    // Find the data index of this element in the original matrix.
                    // Note the reversal of the roles of k an j.
                    let transposed_idx =
                        i * lead_stride + k * strides[num_dims - 2] + j * strides[num_dims - 1];
                    new_data.push(self.data[transposed_idx].clone());
                }
            }
        }
        return TensorImpl {
            shape: new_shape,
            data: new_data,
        };
    }

    fn matmul(&self, other: &Self) -> Result<Self, Self::TensorError> {
        return self.matmul_transpose(&other.transpose());
    }

    /// Interleaves last dimension of the input tensors and concatenates them along the last dimension.
    fn concat(
        self: &Self,
        other: &Self,
        dim: usize,
    ) -> Result<Self, <Self as Tensor<E>>::TensorError> {
        if self.num_dims() != other.num_dims() {
            return Err(Error::msg("Tensors must have the same number of dimensions").into());
        }
        for i in 0..self.shape.len() {
            if i != dim && self.shape[i] != other.shape[i] {
                return Err(Error::msg(
                    "Tensors must have the same shape except for the concatenation dimension",
                )
                .into());
            }
        }

        let n_dims = self.num_dims();
        let self_chunk_dim = self
            .shape
            .iter()
            .rev()
            .take(n_dims - dim)
            .product::<usize>();
        let other_chunk_dim = other
            .shape
            .iter()
            .rev()
            .take(n_dims - dim)
            .product::<usize>();

        let new_data: Vec<E> = self
            .data
            .chunks(self_chunk_dim)
            .zip(other.data.chunks(other_chunk_dim)) // yields items like (&[1.0, 2.0, 3.0], &[7.0, 8.0, 9.0])
            .flat_map(|(a, b)| a.into_iter().chain(b)) // chains to produce iterators like [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
            // TODO: consider adding a bound on copy
            // .copied()
            .cloned() // &f64 -> f64, optional
            .collect();

        // Convert output_vec into a tensor with required output shape
        let mut new_shape = self.shape.clone();
        new_shape[dim] = self.shape[dim] + other.shape[dim];
        return Ok(Self {
            shape: new_shape,
            data: new_data,
        });
    }

    /// Sum across one or more dimensions (eg. row-wise sum for a 2D matrix resulting in a "column
    /// vector")
    fn dim_sum(&self, dims: Vec<usize>) -> Self {
        // naive implementation, just looping over the dimensions
        let mut result = self.clone();
        for dim in dims {
            result = result.single_dim_sum(dim);
        }
        return result;
    }

    // TODO(mhauru) This should return something like a view, which references the same data but is
    // a new object. I don't know how to do that though.
    fn reshape(&mut self, new_shape: Vec<usize>) {
        let num_els = self.num_elements();
        let new_num_els = num_elements_from_shape(&new_shape);
        // println!("Num els {}", num_els);
        // println!("New num els{}", new_num_els);
        // println!("Shape: {:?}", self.shape());
        if self.num_elements() != num_elements_from_shape(&new_shape) {
            panic!("The number of elements in the new shape does not match the number of elements in the original shape.");
        }
        self.shape = new_shape;
    }
}

impl<E> TensorImpl<E>
where
    E: Element,
{
    pub fn get_data(&self) -> &Vec<E> {
        self.data.as_ref()
    }


    // TODO (asmith) There is loads of copy-and-paste code from the
    // `single_dim_sum` method. Ideally this should be refactored
    pub fn slice(&self, dim: usize, idx: usize) -> Result<Self, &str> {
        if dim >= self.shape.len() {
            return Err("The provided dimension is out of bounds.");
        }

        if idx >= self.shape[dim] {
            return Err("The provided index is out of bounds.");
        }


        let leading_dims = self.shape.iter().take(dim).into_iter().product::<usize>();
        let trailing_dims = self
            .shape
            .iter()
            .skip(dim + 1)
            .into_iter()
            .product::<usize>();

        println!("leading_dims: {}", leading_dims);
        println!("self.shape[dim]: {}", self.shape[dim]);
        println!("trailing_dims: {}", trailing_dims);

        let mut output_shape = self.shape.clone();
        output_shape[dim] = 1;


        let mut slice: Vec<E> = Vec::new();

        // Outer loop needs to iterate over the size of the new shape
        for lead_idx in 0..leading_dims {
            for trail_idx in 0..trailing_dims {
                for slice_idx in 0..self.shape[dim] {
                    if slice_idx == idx {
                        let idx = lead_idx * self.shape[dim] * trailing_dims
                            + slice_idx * trailing_dims
                            + trail_idx;
                        slice.push(self.data[idx].clone());
                    }
                }
            }
        }

        // println!("output_shape: {:?}", output_shape);
        // println!("slice: {:?}", slice);
        Ok(TensorImpl::from_vec(&output_shape, &slice).unwrap())

    }

    ///// Sum across a single dimensions (eg. row-wise sum for a 2D matrix resulting in a "column
    ///// vector")
    fn single_dim_sum(&self, dim: usize) -> Self {
        if dim >= self.shape.len() {
            panic!("The provided dimension is out of bounds.");
        }

        let leading_dims = self.shape.iter().take(dim).into_iter().product::<usize>();
        let trailing_dims = self
            .shape
            .iter()
            .skip(dim + 1)
            .into_iter()
            .product::<usize>();

        let mut output_shape = self.shape.clone();
        output_shape[dim] = 1;
        let output_size = output_shape.iter().product::<usize>();

        let mut dim_sum: Vec<E> = Vec::new();

        // Outer loop needs to iterate over the size of the new shape
        for lead_idx in 0..leading_dims {
            for trail_idx in 0..trailing_dims {
                let mut sum: E = E::zero();
                for summing_idx in 0..self.shape[dim] {
                    let idx = lead_idx * self.shape[dim] * trailing_dims
                        + summing_idx * trailing_dims
                        + trail_idx;
                    // sum += self.data[idx].clone();
                    sum = sum + self.data[idx].clone();
                }
                dim_sum.push(sum);
            }
        }

        TensorImpl {
            shape: output_shape,
            data: dim_sum,
        }
    }
}

impl<E: RealElement> Exp for TensorImpl<E> {
    fn exp(self) -> Self {
        let new_data = self.data.iter().map(|x| x.clone().exp()).collect();
        TensorImpl {
            shape: self.shape,
            data: new_data,
        }
    }
}

impl<E: RealElement> Pow<E> for TensorImpl<E> {
    fn pow(self, exp: E) -> Self {
        let new_data = self
            .data
            .iter()
            .map(|x| x.clone().pow(exp.clone()))
            .collect();
        TensorImpl {
            shape: self.shape,
            data: new_data,
        }
    }
}

impl<E: RealElement> Ln for TensorImpl<E> {
    fn ln(self) -> Self {
        let new_data = self.data.iter().map(|x| x.clone().ln()).collect();
        TensorImpl {
            shape: self.shape,
            data: new_data,
        }
    }
}

impl<E: RealElement> RealTensor<E> for TensorImpl<E> {
    fn softmax(&self, dim: usize) -> Self {
        let t_small = E::from(f64::EPSILON);
        let max = self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let data_exp = (self.clone() - max.clone()).exp();
        let data_sum = data_exp.dim_sum(vec![dim]);

        let new_data = data_exp / (data_sum + t_small);
        TensorImpl {
            shape: self.shape.clone(),
            data: new_data.data,
        }
    }

    fn fill_from_f64(shape: Vec<usize>, data: f64) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn make_random_f64_tensor(
        rng: &mut rand::rngs::ThreadRng,
        shape: Vec<usize>,
    ) -> TensorImpl<f64> {
        let data: Vec<f64> = (0..num_elements_from_shape(&shape))
            .map(|_| rng.gen::<f64>())
            .collect();
        TensorImpl::from_vec(&shape, &data).unwrap()
    }

    fn make_range_tensor(shape: Vec<usize>) -> TensorImpl<i32> {
        let num_elements: i32 = num_elements_from_shape(&shape).try_into().unwrap();
        let data = (0..num_elements).collect::<Vec<i32>>();
        TensorImpl { shape, data }
    }

    #[test]
    fn test_from_vec() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let maybe_tensor = TensorImpl::from_vec(&shape, &data);
        let tensor = maybe_tensor.unwrap();

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_from_vec_invalid_params() {
        // The length of the `data` Vec is not that implied by the `shape` Vec
        let shape = vec![2, 3];
        let data = vec![1, 2];
        let maybe_tensor = TensorImpl::from_vec(&shape, &data);
        assert!(maybe_tensor.is_err());
        let err = maybe_tensor.unwrap_err();

        assert!(err.to_string().contains(
            "The length of the `data` param does not match the values of the `shape` param"
        ));
    }

    #[test]
    fn test_shape() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data);

        assert_eq!(tensor.unwrap().shape(), shape);
    }

    #[test]
    fn test_transpose_1d_tensor() {
        let shape = vec![2];
        let data = vec![1, 2];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        let transposed = tensor.transpose();
        assert_eq!(transposed.shape(), shape);
        // Transposing a 1D tensor should return the original tensor
        assert_eq!(transposed.data, data);
    }

    #[test]
    fn test_transpose_2d_tensor() {
        // Case 1
        let shape1 = vec![2, 3];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();
        let transposed1 = tensor1.transpose();
        assert_eq!(transposed1.shape(), vec![3, 2]);
        // Transposing twice should return the original tensor
        let transposed_twice1 = transposed1.transpose();
        assert_eq!(transposed_twice1.shape(), shape1);
        assert_eq!(transposed_twice1.data, data1);

        // Case 2
        let shape2 = vec![2, 2];
        let original_data2 = vec![1, 2, 3, 4];
        let tensor2 = TensorImpl::from_vec(&shape2, &original_data2).unwrap();
        let transposed2 = tensor2.transpose();
        assert_eq!(transposed2.shape(), vec![2, 2]);

        let expected_data2 = vec![1, 3, 2, 4];
        assert_eq!(transposed2.data, expected_data2);
    }

    #[test]
    fn test_transpose_3d_tensor() {
        let tensor = make_range_tensor(vec![3, 4, 5]);
        let transposed = tensor.transpose();
        let expected_shape = vec![3, 5, 4];
        assert_eq!(transposed.shape(), expected_shape);

        // The data should be different from the original tensor
        assert_ne!(transposed.data, tensor.data);

        // Transposing twice should return the original tensor
        let transposed_twice = transposed.transpose();
        assert_eq!(transposed_twice.shape(), tensor.shape);
        assert_eq!(transposed_twice.data, tensor.data);
    }

    #[test]
    fn test_at() {
        let tensor = make_range_tensor(vec![3, 2, 2]);
        assert_eq!(*tensor.at(vec![0, 0, 0]).unwrap(), 0);
        assert_eq!(*tensor.at(vec![2, 1, 1]).unwrap(), 11);
        assert_eq!(tensor.at(vec![2, 1, 2]), None);
        assert_eq!(*tensor.at(vec![1, 1, 0]).unwrap(), 6);
        assert_eq!(*tensor.at(vec![1, 0, 1]).unwrap(), 5);
    }

    #[test]
    fn test_at_mut() {
        let mut tensor = make_range_tensor(vec![3, 2, 2]);
        let index = vec![2, 0, 1];
        let element = tensor.at_mut(index.clone()).unwrap();
        *element = 100;
        assert_eq!(*tensor.at(index.clone()).unwrap(), 100);
    }

    // Addition
    #[test]
    fn test_adding_tensors() {
        let shape = vec![2, 3];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40, 50, 60];
        let tensor1 = TensorImpl::from_vec(&shape, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape, &data2).unwrap();

        let tensor3 = tensor1 + tensor2;
        assert_eq!(tensor3.data, vec![11, 22, 33, 44, 55, 66]);
    }

    #[test]
    fn test_adding_tensors_wrong_shapes() {
        let shape1 = vec![2, 3];
        let shape2 = vec![2, 2];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        assert!(std::panic::catch_unwind(|| tensor1 + tensor2).is_err());
    }

    #[test]
    fn test_adding_tensors_and_scalars() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let tensor2 = tensor + 10;
        assert_eq!(tensor2.data, vec![11, 12, 13, 14, 15, 16]);
    }

    // Element-wise multiplication
    #[test]
    fn test_multiplying_tensors() {
        let shape = vec![2, 3];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40, 50, 60];
        let tensor1 = TensorImpl::from_vec(&shape, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape, &data2).unwrap();

        let tensor3 = tensor1 * tensor2;
        assert_eq!(tensor3.data, vec![10, 40, 90, 160, 250, 360]);
    }

    #[test]
    fn test_multiplying_tensors_wrong_shapes() {
        let shape1 = vec![2, 3];
        let shape2 = vec![2, 2];
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![10, 20, 30, 40];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        assert!(std::panic::catch_unwind(|| tensor1 * tensor2).is_err());
    }

    #[test]
    fn test_multiplying_tensors_and_scalars() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let tensor2 = tensor * 10;
        assert_eq!(tensor2.data, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_div_tensor_by_scalar() {
        let shape = vec![2, 3];
        let data = vec![10, 20, 30, 40, 50, 60];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let tensor2 = tensor.clone() / 10;
        assert_eq!(tensor2.data, vec![1, 2, 3, 4, 5, 6]);

        // This line should panic, because division by zero is not allowed
        let _tensor3 = tensor.clone() / 0;
    }

    #[test]
    fn test_single_dim_sum() {
        let shape = vec![2, 2, 2];
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let expected_depth_sum = vec![6, 8, 10, 12];
        let expected_depth_shape = vec![1, 2, 2];
        let actual_depth_sum = tensor.single_dim_sum(0);
        assert_eq!(actual_depth_sum.data, expected_depth_sum);
        assert_eq!(actual_depth_sum.shape, expected_depth_shape);

        let expected_col_sum = vec![4, 6, 12, 14];
        let expected_col_shape = vec![2, 1, 2];
        let actual_col_sum = tensor.single_dim_sum(1);
        assert_eq!(actual_col_sum.data, expected_col_sum);
        assert_eq!(actual_col_sum.shape, expected_col_shape);

        let expected_row_sum = vec![3, 7, 11, 15];
        let expected_row_shape = vec![2, 2, 1];
        let actual_row_sum = tensor.single_dim_sum(2);
        assert_eq!(actual_row_sum.data, expected_row_sum);
        assert_eq!(actual_row_sum.shape, expected_row_shape);
    }

    #[test]
    fn test_multiple_dim_sum() {
        let tensor = make_range_tensor(vec![2, 3, 4, 5]);
        // Sum over dimensions 1 and 3
        // The result should have shape [2, 1, 4, 1]
        // The result should not be dependent of the order of the dimensions
        let expected_shape = vec![2, 1, 4, 1];

        let actual_sum_fwd = tensor.dim_sum(vec![1, 3]);
        let actual_sum_bwd = tensor.dim_sum(vec![3, 1]);

        assert_eq!(actual_sum_fwd.shape, expected_shape);
        assert_eq!(actual_sum_bwd.shape, expected_shape);
        assert_eq!(actual_sum_fwd.data, actual_sum_bwd.data);
    }

    #[test]
    fn test_matmul() {
        let shape1 = vec![3, 2];
        let data1 = vec![
            0.3482904331369514,
            0.6655930709447386,
            0.8870176843052003,
            0.5022740751412192,
            -0.7230098637226343,
            -0.25526520068994935,
        ];
        let tensor1 = TensorImpl::from_vec(&shape1, &data1).unwrap();

        let shape2 = vec![2, 2];
        let data2 = vec![
            -1.7175862448910375,
            -1.5269576121156156,
            0.056288722445149646,
            -0.4679238780318657,
        ];
        let tensor2 = TensorImpl::from_vec(&shape2, &data2).unwrap();

        // The result of multiplying the above two, computed with NumPy.
        let shape_expected = vec![3, 2];
        let data_expected = vec![
            -0.5607534735513462,
            -0.8432716190531137,
            -1.4952570075306946,
            -1.5894644382059397,
            1.2274632448189984,
            1.223450097679374,
        ];

        let result = tensor1.matmul(&tensor2).unwrap();
        assert_eq!(result.shape, shape_expected);
        assert_eq!(result.data, data_expected);

        // Check that A^T * B^T = (B * A)^T
        let shape3 = vec![2, 2];
        let data3 = vec![
            0.7670520197177377,
            0.14770389167309558,
            0.3166885668777075,
            0.326126502112053,
        ];
        let tensor3 = TensorImpl::from_vec(&shape3, &data3).unwrap();

        let lhs = tensor2.transpose().matmul(&tensor3.transpose()).unwrap();
        let rhs = (tensor3.matmul(&tensor2).unwrap()).transpose();
        assert_eq!(lhs, rhs)
    }

    #[test]
    fn test_matmul_multidim() {
        let mut rng = rand::thread_rng();
        let tensor1 = make_random_f64_tensor(&mut rng, vec![5, 4, 3, 2]);
        let tensor2 = make_random_f64_tensor(&mut rng, vec![2, 6]);
        let result = tensor1.matmul(&tensor2).unwrap();
        let shape_expected = vec![5, 4, 3, 6];
        assert_eq!(result.shape, shape_expected);
    }

    #[test]
    fn test_fill_with_clone() {
        let shape = vec![2, 3];
        let element = 10;
        let tensor = TensorImpl::fill_with_clone(shape, element);

        assert_eq!(tensor.data, vec![10, 10, 10, 10, 10, 10]);
    }

    #[test]
    fn test_into_iter() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        assert_eq!(tensor.into_iter().collect::<Vec<i32>>(), data);
    }

    #[test]
    fn test_to_vec() {
        let shape = vec![2, 3];
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        assert_eq!(Vec::<i32>::from(tensor), data);
    }

    #[test]
    fn test_element_exp() {
        let shape = vec![2, 1];
        let data = vec![1.0, 2.0];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        let expected_data = vec![2.718281828459045, 7.38905609893065];
        let tensor_exp = tensor.exp();
        assert_eq!(tensor_exp.data, expected_data);
    }

    #[test]
    fn test_element_pow() {
        let shape = vec![2, 1];
        let data = vec![1.0, 2.0];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        let expected_data = vec![1.0, 4.0];
        let tensor_pow = tensor.pow(2.0);
        assert_eq!(tensor_pow.data, expected_data);
    }

    #[test]
    fn test_element_ln() {
        let shape = vec![2, 1];
        let data = vec![2.718281828459045, 7.38905609893065];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
        let expected_data = vec![1.0, 2.0];
        let tensor_ln = tensor.ln();
        assert_eq!(tensor_ln.data, expected_data);
    }

    #[test]
    fn test_slice() {
        // Test dim is too large
        {
            let shape = vec![2, 2];
            let data = vec![1, 2, 3, 4];
            let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
    
            let slice = tensor.slice(5, 1);
            assert!(slice.is_err());
            assert_eq!(slice.err(), Some("The provided dimension is out of bounds."));
        }

        // Test idx is too large
        {
            let shape = vec![2, 2];
            let data = vec![1, 2, 3, 4];
            let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
    
            let slice = tensor.slice(1, 5);
            assert!(slice.is_err());
            assert_eq!(slice.err(), Some("The provided index is out of bounds."));
        }

        // Test working case
        let shape = vec![2, 2, 2];
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let tensor = TensorImpl::from_vec(&shape, &data).unwrap();

        let maybe_slice = tensor.slice(2,1);
        let expected_shape = vec![2, 2, 1];
        let expected_data = vec![2, 4, 6, 8];

        assert!(maybe_slice.is_ok());
        let slice = maybe_slice.unwrap();

        assert_eq!(slice.shape(), expected_shape);
        assert_eq!(slice.data, expected_data);
    }

    #[test]
    fn test_concat() {
        let mut rng = rand::thread_rng();

        {
            // Concat over last dim
            let tensor1 = make_random_f64_tensor(&mut rng, vec![5, 4, 3, 2]);
            let tensor2 = make_random_f64_tensor(&mut rng, vec![5, 4, 3, 3]);
            let result = tensor1.concat(&tensor2, 3).unwrap();
            let shape_expected = vec![5, 4, 3, 5];
            assert_eq!(result.shape, shape_expected);
            // Check that the result has the same elements as the original, just in a different order.
            let mut sorted_expected = tensor1.data.clone();
            sorted_expected.extend(tensor2.data.clone());
            sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut sorted_elements = result.data.clone();
            sorted_elements.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(sorted_elements, sorted_expected);
        }

        {
            // Concat over first dim
            let tensor1 = make_random_f64_tensor(&mut rng, vec![5, 4, 3, 2]);
            let tensor2 = make_random_f64_tensor(&mut rng, vec![4, 4, 3, 2]);
            let result = tensor1.concat(&tensor2, 0).unwrap();
            let shape_expected = vec![9, 4, 3, 2];
            assert_eq!(result.shape, shape_expected);
            // Check that the result has the same elements as the original, just in a different order.
            let mut sorted_expected = tensor1.data.clone();
            sorted_expected.extend(tensor2.data.clone());
            sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut sorted_elements = result.data.clone();
            sorted_elements.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(sorted_elements, sorted_expected);
        }

        {
            // Concat over middle dim
            let tensor1 = make_random_f64_tensor(&mut rng, vec![5, 4, 2, 2]);
            let tensor2 = make_random_f64_tensor(&mut rng, vec![5, 4, 3, 2]);
            let result = tensor1.concat(&tensor2, 2).unwrap();
            let shape_expected = vec![5, 4, 5, 2];
            assert_eq!(result.shape, shape_expected);
            // Check that the result has the same elements as the original, just in a different order.
            let mut sorted_expected = tensor1.data.clone();
            sorted_expected.extend(tensor2.data.clone());
            sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut sorted_elements = result.data.clone();
            sorted_elements.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(sorted_elements, sorted_expected);
        }
    }

    #[test]
    /// Test that concat(A * B, A * C, 2) == A * concat(B, C, 2) and
    /// concat(B * A, C * A, 2) == concat(B, C, 2) * A.
    fn test_concat_matmul() {
        let mut rng = rand::thread_rng();

        {
            let tensor1 = make_random_f64_tensor(&mut rng, vec![3, 2]);
            let tensor2 = make_random_f64_tensor(&mut rng, vec![3, 4]);
            let tensor3 = make_random_f64_tensor(&mut rng, vec![3, 3]);
            let result1 = tensor3
                .matmul(&tensor1.concat(&tensor2, 1).unwrap())
                .unwrap();
            let result2 = tensor3
                .matmul(&tensor1)
                .unwrap()
                .concat(&tensor3.matmul(&tensor2).unwrap(), 1)
                .unwrap();
            assert_eq!(result1.shape, result2.shape);
            assert_eq!(result1.data, result2.data);
        }

        {
            let tensor1 = make_random_f64_tensor(&mut rng, vec![3, 2]);
            let tensor2 = make_random_f64_tensor(&mut rng, vec![4, 2]);
            let tensor3 = make_random_f64_tensor(&mut rng, vec![2, 2]);
            let result1 = &tensor1
                .concat(&tensor2, 0)
                .unwrap()
                .matmul(&tensor3)
                .unwrap();
            let result2 = tensor1
                .matmul(&tensor3)
                .unwrap()
                .concat(&tensor2.matmul(&tensor3).unwrap(), 0)
                .unwrap();
            assert_eq!(result1.shape, result2.shape);
            assert_eq!(result1.data, result2.data);
        }

        {
            let tensor1 = make_random_f64_tensor(&mut rng, vec![2, 3, 2]);
            let tensor2 = make_random_f64_tensor(&mut rng, vec![4, 3, 2]);
            let tensor3 = make_random_f64_tensor(&mut rng, vec![2, 2]);
            let result1 = &tensor1
                .concat(&tensor2, 0)
                .unwrap()
                .matmul(&tensor3)
                .unwrap();
            let result2 = tensor1
                .matmul(&tensor3)
                .unwrap()
                .concat(&tensor2.matmul(&tensor3).unwrap(), 0)
                .unwrap();
            assert_eq!(result1.shape, result2.shape);
            assert_eq!(result1.data, result2.data);
        }
    }

    #[test]
    fn test_softmax() {
        {
            let shape = vec![2, 3, 4, 5];
            let data = (0u32..num_elements_from_shape(&shape) as u32)
                .map(f64::from)
                .collect::<Vec<f64>>();

            let tensor = TensorImpl::from_vec(&shape, &data).unwrap();
            let dim_to_softmax = 1;
            let result = tensor.softmax(dim_to_softmax);

            // Shape should be the unchanged
            assert_eq!(result.shape(), shape.clone());

            // All of the elements within the result should be within the range [0, 1]
            for element in result.data.iter() {
                assert!(*element >= 0.0 && *element <= 1.0);
            }

            // Calling dim_sum on the result of softmaxed should give a tensor with all 1s
            let dim_sum_of_result = result.dim_sum(vec![dim_to_softmax]);

            for element in dim_sum_of_result.data.iter() {
                // The value should be close to 1, but not exactly 1 due to floating point errors
                assert!((element - 1.0f64).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_add_broadcast() {
        {
            let tensor1 = make_range_tensor(vec![2, 3, 2]);
            let tensor2 = make_range_tensor(vec![1, 3, 2]);
            let result1 = tensor1.clone() + tensor2.clone();
            let result2 = tensor1 + tensor2;
            let expected_shape = vec![2, 3, 2];
            let expected_data = vec![0, 2, 4, 6, 8, 10, 6, 8, 10, 12, 14, 16];
            let expected_result = TensorImpl::from_vec(&expected_shape, &expected_data).unwrap();
            assert_eq!(result1, expected_result);
            assert_eq!(result2, expected_result);
        }

        {
            let tensor1 = make_range_tensor(vec![2, 3, 2]);
            let tensor2 = make_range_tensor(vec![2, 1, 2]);
            let result1 = tensor1.clone() + tensor2.clone();
            let result2 = tensor1 + tensor2;
            let expected_shape = vec![2, 3, 2];
            let expected_data = vec![0, 2, 2, 4, 4, 6, 8, 10, 10, 12, 12, 14];
            let expected_result = TensorImpl::from_vec(&expected_shape, &expected_data).unwrap();
            assert_eq!(result1, expected_result);
            assert_eq!(result2, expected_result);
        }

        {
            let tensor1 = make_range_tensor(vec![2, 3, 2]);
            let tensor2 = make_range_tensor(vec![2, 3, 1]);
            let result1 = tensor1.clone() + tensor2.clone();
            let result2 = tensor1 + tensor2;
            let expected_shape = vec![2, 3, 2];
            let expected_data = vec![0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16];
            let expected_result = TensorImpl::from_vec(&expected_shape, &expected_data).unwrap();
            assert_eq!(result1, expected_result);
            assert_eq!(result2, expected_result);
        }

        {
            let tensor1 = make_range_tensor(vec![2, 3, 2]);
            let tensor2 = make_range_tensor(vec![2, 1, 1]);
            let result1 = tensor1.clone() + tensor2.clone();
            let result2 = tensor1 + tensor2;
            let expected_shape = vec![2, 3, 2];
            let expected_data = vec![0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12];
            let expected_result = TensorImpl::from_vec(&expected_shape, &expected_data).unwrap();
            assert_eq!(result1, expected_result);
            assert_eq!(result2, expected_result);
        }

        {
            let tensor1 = make_range_tensor(vec![2, 1, 3]);
            let tensor2 = make_range_tensor(vec![1, 2, 1]);
            let result1 = tensor1.clone() + tensor2.clone();
            let result2 = tensor1 + tensor2;
            let expected_shape = vec![2, 2, 3];
            let expected_data = vec![0, 1, 2, 1, 2, 3, 3, 4, 5, 4, 5, 6];
            let expected_result = TensorImpl::from_vec(&expected_shape, &expected_data).unwrap();
            assert_eq!(result1, expected_result);
            assert_eq!(result2, expected_result);
        }
    }
}
