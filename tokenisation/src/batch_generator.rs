use crate::tokeniser::Tokeniser;
use autodiff::node::Node;
use interfaces::tensors::Tensor;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::iter::zip;
use tensors::TensorImpl;
// Making a batch generator
pub struct BatchGenerator {
    rng: ChaCha8Rng,
    text: String,
    tokeniser: Tokeniser,
    tokens: Vec<usize>,
    chunk_len: usize,
    batch_size: usize,
    vocab_size: usize,
}

impl BatchGenerator {
    pub fn new(text: String, chunk_len: usize, batch_size: usize, seed: u64) -> BatchGenerator {
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let tokeniser = Tokeniser::new(&text);
        let tokens = tokeniser.encode(&text);
        let vocab_size = tokeniser.vocab_size();
        BatchGenerator {
            rng,
            text,
            tokeniser,
            tokens,
            chunk_len,
            batch_size,
            vocab_size,
        }
    }

    fn sample(&mut self) -> TrainingExample {
        let rand_idx = self
            .rng
            .gen_range(0..self.tokens.len() - (self.chunk_len + 1));
        let input = self.tokens[rand_idx..(rand_idx + self.chunk_len)].to_vec();
        let target = self.tokens[(rand_idx + 1)..(rand_idx + self.chunk_len + 1)].to_vec();

        TrainingExample { input, target }
    }

    pub fn sample_batch(&mut self) -> (TensorImpl<Node<f64>>, TensorImpl<Node<f64>>) {
        let (mut x_tensor, mut y_tensor) = (Vec::new(), Vec::new());
        for _ in 0..self.batch_size {
            let sample = self.sample();
            for (x_token, y_token) in zip(sample.input, sample.target) {
                // let mut one_hot_x: Vec<Node<f64>> = vec![Node::from(0.0); self.vocab_size];
                let mut one_hot_y: Vec<Node<f64>> = vec![Node::from(0.0); self.vocab_size];
                // one_hot_x[x_token] = Node::from(1.0);
                one_hot_y[y_token] = Node::from(1.0);
                x_tensor.push(Node::from(x_token));
                y_tensor.extend(one_hot_y);
            }
        }

        let x = TensorImpl::from_vec(
            // &vec![self.batch_size, self.chunk_len, self.vocab_size],
            &vec![self.batch_size, self.chunk_len, 1],
            &x_tensor,
        )
        .unwrap();
        let y = TensorImpl::from_vec(
            &vec![self.batch_size, self.chunk_len, self.vocab_size],
            &y_tensor,
        )
        .unwrap();
        (x, y)
    }
}

pub struct TrainingExample {
    pub input: Vec<usize>,
    pub target: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_sample() {
        let seed = 0;
        let text = "this is some dummy text".into();
        let chunk_len = 4;
        let batch_size = 1;
        let mut batch_gen = BatchGenerator::new(text, chunk_len, batch_size, seed);

        let example = batch_gen.sample();
        println!("input: {:?}", example.input);
        println!("target: {:?}", example.target);
        assert_eq!(example.input[1], example.target[0]);
        assert_eq!(example.input[2], example.target[1]);
        assert_eq!(example.input[3], example.target[2]);
    }

    #[test]
    fn generate_batch() {
        let seed = 0;
        let text = "this is some dummy text".into();
        let chunk_len = 4;
        let batch_size = 1;
        let mut batch_gen = BatchGenerator::new(text, chunk_len, batch_size, seed);

        let (x, y) = batch_gen.sample_batch();
    }
}
