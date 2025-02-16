use ndarray::Array2;

pub fn cross_attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    let d_k = q.shape()[1] as f32;
    let mut scores = q.dot(&k.t()) / d_k.sqrt();

    for mut row in scores.rows_mut() {
        let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|x| x - max);
        row.mapv_inplace(f32::exp);
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    }

    scores.dot(v)
}
