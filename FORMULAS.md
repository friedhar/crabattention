# Formulas Reference

$\text{casual attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
$\text{casual attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V$
