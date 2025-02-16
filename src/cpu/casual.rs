use ndarray::{Array, Array2, IntoDimension};

// ref: Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V

// ref: softmax(x) =
// 1.      find max(x)
// 2.      normalize each value of x by subtracting max(x)
// 3.      exp(x), in order to both have exponential propeties + make sure x_i >= 0 for all x
// 4.      exp(x) / sum(exp(x)) normalize, make sure sum ~= 1.0

fn causal_mask(seq_len: usize) -> Array2<f32> {
    Array::from_shape_fn((seq_len, seq_len), |(i, j)| if j > i { -1e9 } else { 0.0 })
}

pub fn causal_attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    let d_k = q.shape()[1] as f32;
    let scores = q.dot(&k.t()) / d_k.sqrt();

    let seq_len = q.shape()[0];
    let mask = causal_mask(seq_len);

    let mut masked_scores = scores + mask;

    for mut row in masked_scores.rows_mut() {
        let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|x| x - max);
        row.mapv_inplace(f32::exp);
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    }

    masked_scores.dot(v)
}

#[cfg(test)]
mod tests {
    use crate::cpu::casual::causal_attention;

    #[test]
    fn test0() {
        // Example usage
        let q = Array2::from(vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let k = q.clone();
        let v = q.clone();

        let output = causal_attention(&q, &k, &v);
        println!("Causal attention output:\n{:.2}", output);
    }
}
