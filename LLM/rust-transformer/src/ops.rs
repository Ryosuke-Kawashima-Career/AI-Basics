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

fn relu(x: &Array<f32, Ix1>) -> Array<f32, Ix1> {
    let result: Array<f32, Ix1> = x.mapv(|val| if val < 0.0 { 0.0 } else { val });
    result
}

fn gelu(x: &Array<f32, Ix1>) -> Array<f32, Ix1> {
    /*Calculates Gaussian Error Linear Unit*/
    let result: Array<f32, Ix1> = x.mapv(|val| {
        0.5 * val
            * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (val + 0.044715 * val.powi(3))).tanh())
    });
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
    #[test]
    fn test_relu() {
        let a: Array<f32, Ix1> = array![-1.0f32, 0.0f32, 1.0f32];
        let res = relu(&a);
        let expected: Array<f32, Ix1> = array![0.0f32, 0.0f32, 1.0f32];
        res.iter()
            .zip(expected.iter())
            .for_each(|(x, y)| assert!((x - y).abs() < 1e-5));
    }
    #[test]
    fn test_gelu() {
        let a: Array<f32, Ix1> = array![-1.0f32, 0.0f32, 1.0f32];
        let res = gelu(&a);
        let expected: Array<f32, Ix1> = array![-0.158808f32, 0.0f32, 0.841192f32];
        res.iter()
            .zip(expected.iter())
            .for_each(|(x, y)| assert!((x - y).abs() < 1e-3));
    }
}
