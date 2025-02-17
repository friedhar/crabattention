use ndarray::Array2;

pub fn self_attention(
    x: &Array2<f32>,
    w_q: &Array2<f32>,
    w_k: &Array2<f32>,
    w_v: &Array2<f32>,
) -> Array2<f32> {
    let q = x.dot(w_q);
    let k = x.dot(w_k);
    let v = x.dot(w_v);

    let d_k = x.shape()[1] as f32;
    let mut scores = q.dot(&k.t()) / d_k.sqrt();

    for mut row in scores.rows_mut() {
        let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|x| x - max);
        row.mapv_inplace(f32::exp);
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    }

    scores.dot(&v)
}
