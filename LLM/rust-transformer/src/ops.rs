use ndarray::prelude::*;
use ndarray::{Array, Ix1};
fn softmax(x: &Array<f32, Ix1>) -> Array<f32, Ix1> {
    let max_x: f32 = *x
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .expect("Expected non-empty array");
    let exps = x.mapv(|val| (val - max_x).exp());
    let sum_exps: f32 = exps.sum();
    let result = exps.mapv(|val| val / sum_exps);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_softmax() {
        let a: Array<f32, Ix1> = array![-1.0f32, 0.0f32, 1.0f32];
        let res = softmax(&a);
        let expected: Array<f32, Ix1> = array![0.09003057, 0.24472847, 0.66524096];
        res.iter()
            .zip(expected.iter())
            .for_each(|(x, y)| assert!((x - y).abs() < 1e-5));
    }
}
