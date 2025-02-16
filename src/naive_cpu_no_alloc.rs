fn matmul(a: &[&[f32]], b: &[&[f32]]) -> Vec<Vec<f32>> {
    let rows = a.len();
    let cols = b[0].len();
    let common = b.len();
    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            for k in 0..common {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn softmax(scores: &mut [&mut [f32]]) {
    for row in scores.iter_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (i, val) in row.iter_mut().enumerate() {
            *val = exps[i] / sum;
        }
    }
}

fn causal_attention(q: &[&[f32]], k: &[&[f32]], v: &[&[f32]]) -> Vec<Vec<f32>> {
    let scale = (k.len() as f32).sqrt().recip(); // 1/sqrt(d_k)
    let mut scores = matmul(q, &transpose(k)); // Q @ K^T

    let n = scores.len();
    for i in 0..n {
        for j in (i + 1)..n {
            scores[i][j] = f32::NEG_INFINITY;
        }
        for j in 0..n {
            scores[i][j] *= scale;
        }
    }

    softmax(&mut scores);
    matmul(&scores, v)
}

fn transpose(matrix: &[&[f32]]) -> Vec<Vec<f32>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    for row in matrix {
        assert_eq!(row.len(), cols, "Inconsistent row lengths in matrix");
    }

    let mut result = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, time::Instant};

    use crate::naive_cpu_no_alloc::{causal_attention, transpose};

    #[test]
    fn t0() {
        let q: &[&[f32]] = &[&[0.1, 0.2], &[0.3, 0.4], &[0.5, 0.6]];
        let k: &[&[f32]] = &[&[0.2, 0.1], &[0.4, 0.3], &[0.6, 0.5]];
        let v: &[&[f32]] = &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];

        let output = causal_attention(&q, &k, &v);

        println!("Output:");
        for row in &output {
            println!("{:?}", row);
        }
    }

    #[test]
    fn benchmark_naive_cpu_constant_size() {
        let q: &[&[f32]] = &[&[0.1, 0.2], &[0.3, 0.4], &[0.5, 0.6]];
        let k: &[&[f32]] = &[&[0.2, 0.1], &[0.4, 0.3], &[0.6, 0.5]];
        let v: &[&[f32]] = &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];

        let mut s_t = Instant::now();
        let n = 33_554_432; // 2^20

        for _ in 0..n {
            black_box(causal_attention(&q, &k, &v));
        }
        let took = s_t.elapsed() / n;
        println!("mean(took): {took:?}")
    }
}
