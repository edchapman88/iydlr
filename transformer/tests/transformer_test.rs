use attention::attention::{El, Te};
use config::Config;
use interfaces::deep_learning::DLModule;
use interfaces::tensors::{RealTensor, Tensor};
use neural_nets::optim::{bce, cce, OptimSGD};
use tokenisation::batch_generator::BatchGenerator;
use transformer::transformer::Transformer;

fn get_config() -> Config {
    Config {
        batch_size: 1,
        seq_len: 8,
        embed_dim: 8,
        // Currently this must be set as the same as the input text
        vocab_size: 12,
        num_head: 4,
        num_blocks: 4,
        seed: 0,
    }
}

#[test]
fn transformer_test() {
    let max_itr = 300;
    let config = &get_config();
    let model = Transformer::new(config);
    let mut batch_gen = BatchGenerator::new(
        "hellotransformer".to_string(),
        config.seq_len,
        config.batch_size,
        config.seed,
    );

    // let out = model.forward(&x).unwrap();
    // let expected_shape = vec![2, 7, 12];
    // let actual_shape = out.shape();
    // assert_eq!(actual_shape, expected_shape);

    let mut optim = OptimSGD::new(0.01, max_itr, model.params());

    for itr in 0..max_itr {
        println!("Iteration: {}", itr);
        let (x, y) = batch_gen.sample_batch();
        println!("x shape: {:?}", x.shape());
        println!("y shape: {:?}", y.shape());
        let pred = model.forward(&x).unwrap();
        let pred_vec: Vec<_> = pred.clone().into();
        println!(
            "{:?}",
            pred_vec
                .into_iter()
                .map(|node| node.val())
                .collect::<Vec<_>>()
        );
        // let loss = cce(&y, &pred);
        // println!(
        //     "loss: {:?}",
        //     loss.clone()
        //         .into_iter()
        //         .map(|node| node.val())
        //         .collect::<Vec<_>>()
        // );

        // optim.zero_grad();
        // println!("Backward...");
        // loss.at(vec![0, 0, 0]).unwrap().clone().backward(1.0);
        // optim.update(itr);
        // println!("Updated optim...");
    }

    // shape (1,B,1)
    // let pred = model.forward(&x).unwrap();
    // println!("class_0: {:?}", cce(y, y_pred));
    // println!(
    //     "truth: {:?}",
    //     y_tensor
    //         .clone()
    //         .into_iter()
    //         .map(|node| node.val())
    //         .collect::<Vec<_>>()
    // );

    // let loss_tensor = bce(y_tensor, class_0);
    // let loss = loss_tensor.dim_sum(vec![1]);
    // assert!(loss.at(vec![0, 0, 0]).unwrap().clone().val() < 0.1_f64)
}
