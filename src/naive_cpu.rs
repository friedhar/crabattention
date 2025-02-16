fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
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

fn softmax(scores: &mut Vec<Vec<f32>>) {
    for row in scores.iter_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (i, val) in row.iter_mut().enumerate() {
            *val = exps[i] / sum;
        }
    }
}

fn causal_attention(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let scale = (k.len() as f32).sqrt().recip(); // 1/sqrt(d_k)
    let mut scores = matmul(q, k); // Q @ K^T

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

fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
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
    use crate::naive_cpu::{causal_attention, transpose};

    #[test]
    fn t0() {
        let q = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let k = vec![vec![0.2, 0.1], vec![0.4, 0.3], vec![0.6, 0.5]];
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let output = causal_attention(&q, &transpose(&k), &v);

        println!("Output:");
        for row in &output {
            println!("{:?}", row);
        }
    }
}
