use interfaces::deep_learning::{DLModule, LinearLayer};
use interfaces::tensors::{Element, Tensor};
use rand::distributions::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Normal;
use std::marker::PhantomData;

pub struct LinLayer<T: Tensor<E>, E: Element> {
    pub w: T,
    pub b: T,
    tensor_element_phantom: PhantomData<E>,
}

impl<T, E> DLModule<T, E> for LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
    type DLModuleError = <T as Tensor<E>>::TensorError;

    fn forward(&self, x: &T) -> Result<T, Self::DLModuleError> {
        let input_shape = x.shape();
        let mut b = self.b.clone();
        // println!("Input shape : {:?}", input_shape);
        // println!("Bias shape : {:?}", self.b.shape());
        // If input has a batch dim, then reshape bias to enable
        // broadcast over batch
        if input_shape.len() > 2 {
            let mut new_shape = vec![1];
            new_shape.extend(b.shape());
            // println!("New bias shape: {:?}", new_shape);
            b.reshape(new_shape);
            // println!("Reshaped bias: {:?}", b.shape());
        }
        Ok(x.clone().matmul(&self.w.clone())? + b)
    }

    fn params(&self) -> Vec<E> {
        let mut res: Vec<E> = self.w.clone().into();
        res.extend(self.b.clone().into());
        res
    }
}

impl<T, E> LinearLayer<T, E> for LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element,
{
}

impl<T, E> LinLayer<T, E>
where
    T: Tensor<E>,
    E: Element + From<f64>,
{
    pub fn new(i_size: usize, o_size: usize, seed: u64) -> Self {
        // He weight initialisation
        // https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        let noise_mean = 0.0;
        let noise_std = f64::sqrt(2.0 / (i_size as f64));
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(noise_mean, noise_std).unwrap();
        let normal_itr = normal.sample_iter(rng);

        let (w_data, b_data): (Vec<E>, Vec<E>) = normal_itr
            .take(o_size * (i_size + 1))
            .map(E::from)
            .enumerate()
            .fold((Vec::new(), Vec::new()), |mut acc, (idx, el)| {
                if idx < o_size * i_size {
                    acc.0.push(el);
                    acc
                } else {
                    acc.1.push(el);
                    acc
                }
            });

        let weights = T::from_vec(&vec![i_size, o_size], &w_data)
            .expect("Ensured data can be arranged into a matrix of the given size.");
        // let bias = T::from_vec(&vec![1_usize, 1_usize, o_size], &b_data)
        let bias = T::from_vec(&vec![1_usize, o_size], &b_data)
            .expect("Ensured data can be arranged into a matrix of the given size.");

        LinLayer {
            w: weights,
            b: bias,
            tensor_element_phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensors::TensorImpl;

    #[test]
    fn construct_lin_layer() {
        let layer: LinLayer<TensorImpl<f64>, f64> = LinLayer::new(2, 1, 0);
    }

    #[test]
    fn three_dim_forward() {
        // Test that the forward method works when the input tensor is 3D
        let layer: LinLayer<TensorImpl<f64>, f64> = LinLayer::new(2, 3, 0);
        let x = TensorImpl::from_vec(&vec![2, 2, 2], &vec![6.0; 8]).unwrap();
        println!("{:?}", x.shape());
        let out = layer.forward(&x).unwrap();
        println!("{:?}", out);
        assert_eq!(out.shape(), vec![2, 2, 3]);
    }

    #[test]
    fn two_dim_forward() {
        // Test that the forward method works when the input tensor is 2D
        let layer: LinLayer<TensorImpl<f64>, f64> = LinLayer::new(2, 3, 0);
        let x = TensorImpl::from_vec(&vec![2, 2], &vec![6.0; 4]).unwrap();
        println!("{:?}", x.shape());
        let out = layer.forward(&x).unwrap();
        println!("{:?}", out);
        assert_eq!(out.shape(), vec![2, 3]);
    }
}
