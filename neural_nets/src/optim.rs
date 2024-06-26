use autodiff::node::Node;
use interfaces::tensors::{RealElement, RealTensor};

pub struct OptimSGD<T> {
    l_rate: f64,
    max_itr: usize,
    params: Vec<T>,
}

impl<T> OptimSGD<T> {
    pub fn new(l_rate: f64, max_itr: usize, params: Vec<T>) -> OptimSGD<T> {
        OptimSGD {
            l_rate,
            max_itr,
            params,
        }
    }
}

impl OptimSGD<Node<f64>> {
    pub fn zero_grad(&mut self) {
        for p in self.params.iter_mut() {
            p.set_grad(0.0)
        }
    }

    pub fn update(&mut self, itr: usize) {
        let mut l_rate = self.l_rate;
        if itr > self.max_itr.saturating_mul(3).saturating_div(4) {
            l_rate *= 0.1;
        }
        for p in self.params.iter_mut() {
            // println!("{:?}", p.grad());
            p.set_val(p.val() + (-l_rate * p.grad().unwrap()))
        }
    }
}

// fn bce<E>(y: E, y_pred: E) -> E
// where
//     E: RealElement + From<f64>,
// {
//     // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]

//     E::from(-1.0)
//         * (y.clone() * (y_pred.clone() + E::from(0.0000001)).ln()
//             + (E::from(1.0) - y) * (E::from(1.0) - (y_pred - E::from(0.0000001))).ln())
// }

/// Binary cross entropy loss function.
pub fn bce<T, E>(y: T, y_pred: T) -> T
where
    T: RealTensor<E>,
    E: RealElement + From<f64>,
{
    // -1 * [ y * (y_pred + 0.0001).ln()    +    (1 - y) * (1 - (y_pred - 0.0001)).ln() ]
    let t_ones = T::fill_with_clone(y.shape(), E::from(1.0));
    T::fill_with_clone(y.shape(), E::from(-1.0))
        * (y.clone() * (y_pred.clone() + E::from(0.0000001)).ln()
            + (t_ones.clone() + (y * E::from(-1.0)))
                * (t_ones + (y_pred + E::from(-0.0000001)) * E::from(-1.0)).ln())
}

/// Categorical (i.e. multi-label) cross entropy loss function.
pub fn cce<T, E>(y: &T, y_pred: &T) -> T
where
    T: RealTensor<E>,
    E: RealElement + From<f64>,
{
    let t_small = E::from(0.00000001);
    let result = (y.clone() * (y_pred.clone() + t_small).ln()).dim_sum(vec![2]);
    let t_negative_ones = E::from(-1.0);
    result * t_negative_ones
}

#[cfg(test)]
mod tests {
    use interfaces::tensors::Tensor;
    use tensors::TensorImpl;

    use super::*;

    #[test]
    fn test_cce() {
        let y_pred = (0..(2 * 2 * 2))
            .into_iter()
            .map(|x| 1_f64 / x as f64)
            .collect::<Vec<f64>>();
        let y = (0..(2 * 2 * 2))
            .into_iter()
            .map(|_| 0 as f64)
            .collect::<Vec<f64>>();

        let shape = vec![2, 2, 2];
        let y_pred = TensorImpl::from_vec(&shape, &y_pred).unwrap();
        let mut y = TensorImpl::from_vec(&shape, &y).unwrap();

        // batch 0.
        let e = y.at_mut(vec![0, 0, 1]).unwrap();
        *e = 1_f64;
        let e = y.at_mut(vec![0, 1, 0]).unwrap();
        *e = 1_f64;

        // batch 1.
        let e = y.at_mut(vec![1, 0, 1]).unwrap();
        *e = 1_f64;
        let e = y.at_mut(vec![1, 1, 0]).unwrap();
        *e = 1_f64;

        let loss = cce(&y, &y_pred);
        println!("{:?}", loss);

        let bce_loss = bce(y, y_pred);
        println!("{:?}", bce_loss);
    }
}
