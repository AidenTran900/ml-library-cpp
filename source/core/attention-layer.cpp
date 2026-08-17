#include "ml_lib/core/attention-layer.h"
#include "ml_lib/core/masking.h"
#include "ml_lib/core/softmax.h"
#include <cmath>
#include <stdexcept>

template<typename T>
AttentionLayer<T>::AttentionLayer(int embed_dim, int num_heads)
    : AttentionLayer(embed_dim, num_heads, num_heads)
{}

template<typename T>
AttentionLayer<T>::AttentionLayer(int embed_dim, int num_heads, int num_kv_heads)
{
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    if (num_heads % num_kv_heads != 0) {
        throw std::invalid_argument("num_heads must be divisible by num_kv_heads");
    }

    this->embed_dim = embed_dim;
    this->num_heads = num_heads;
    this->num_kv_heads = num_kv_heads;
    this->head_dim = embed_dim / num_heads;
    this->kv_dim = num_kv_heads * head_dim;
    this->heads_per_group = num_heads / num_kv_heads;

    W_q = Matrix<T>(embed_dim, embed_dim);
    W_k = Matrix<T>(embed_dim, kv_dim);
    W_v = Matrix<T>(embed_dim, kv_dim);
    W_o = Matrix<T>(embed_dim, embed_dim);

    grad_W_q = Matrix<T>(embed_dim, embed_dim);
    grad_W_k = Matrix<T>(embed_dim, kv_dim);
    grad_W_v = Matrix<T>(embed_dim, kv_dim);
    grad_W_o = Matrix<T>(embed_dim, embed_dim);

    double scale = std::sqrt(2.0 / (embed_dim + embed_dim));
    for (int i = 0; i < embed_dim; i++) {
        for (int j = 0; j < embed_dim; j++) {
            W_q(i, j) = static_cast<T>(((double)rand() / RAND_MAX * 2 - 1) * scale);
            W_o(i, j) = static_cast<T>(((double)rand() / RAND_MAX * 2 - 1) * scale);
        }
        for (int j = 0; j < kv_dim; j++) {
            W_k(i, j) = static_cast<T>(((double)rand() / RAND_MAX * 2 - 1) * scale);
            W_v(i, j) = static_cast<T>(((double)rand() / RAND_MAX * 2 - 1) * scale);
        }
    }
}

template<typename T>
void AttentionLayer<T>::enableRoPE(int max_seq_len, double theta)
{
    rope_enabled = true;
    int half_dim = head_dim / 2;
    rope_cos = Matrix<T>(max_seq_len, half_dim);
    rope_sin = Matrix<T>(max_seq_len, half_dim);

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            double angle = pos / std::pow(theta, (2.0 * i) / head_dim);
            rope_cos(pos, i) = static_cast<T>(std::cos(angle));
            rope_sin(pos, i) = static_cast<T>(std::sin(angle));
        }
    }
}

template<typename T>
void AttentionLayer<T>::applyRoPE(Matrix<T>& Q, Matrix<T>& K, int start_pos)
{
    int seq_len = Q.rows();
    int half_dim = head_dim / 2;

    // Rotate Q (num_heads heads, embed_dim cols)
    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        for (int h = 0; h < num_heads; h++) {
            int head_start = h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                T cos_val = rope_cos(pos, i);
                T sin_val = rope_sin(pos, i);

                T qx = Q(s, head_start + 2 * i);
                T qy = Q(s, head_start + 2 * i + 1);
                Q(s, head_start + 2 * i)     = qx * cos_val - qy * sin_val;
                Q(s, head_start + 2 * i + 1) = qx * sin_val + qy * cos_val;
            }
        }

        // Rotate K (num_kv_heads heads, kv_dim cols)
        for (int h = 0; h < num_kv_heads; h++) {
            int head_start = h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                T cos_val = rope_cos(pos, i);
                T sin_val = rope_sin(pos, i);

                T kx = K(s, head_start + 2 * i);
                T ky = K(s, head_start + 2 * i + 1);
                K(s, head_start + 2 * i)     = kx * cos_val - ky * sin_val;
                K(s, head_start + 2 * i + 1) = kx * sin_val + ky * cos_val;
            }
        }
    }
}

template<typename T>
Matrix<T> AttentionLayer<T>::forward(const Matrix<T>& input)
{
    int seq_len = input.rows();

    input_cache = input;

    Q_cache = input * W_q;
    K_cache = input * W_k;
    V_cache = input * W_v;

    if (rope_enabled) {
        applyRoPE(Q_cache, K_cache, 0);
    }

    Matrix<T> output(seq_len, embed_dim);
    attention_weights_cache.clear();
    attention_weights_cache.resize(num_heads);

    T scale = static_cast<T>(1.0) / static_cast<T>(std::sqrt((double)head_dim));

    for (int h = 0; h < num_heads; h++) {
        int q_start = h * head_dim;
        int kv_h = h / heads_per_group;
        int kv_start = kv_h * head_dim;

        Matrix<T> Q_h(seq_len, head_dim);
        Matrix<T> K_h(seq_len, head_dim);
        Matrix<T> V_h(seq_len, head_dim);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                Q_h(i, j) = Q_cache(i, q_start + j);
                K_h(i, j) = K_cache(i, kv_start + j);
                V_h(i, j) = V_cache(i, kv_start + j);
            }
        }

        // compute attention scores
        Matrix<T> K_h_T = K_h.transpose();
        Matrix<T> scores = Q_h * K_h_T;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                scores(i, j) *= scale;
            }
        }

        // causal masking
        Masking<T> causal_mask(seq_len, seq_len);
        scores = causal_mask.apply(scores);

        // apply softmax
        Matrix<T> attn_weights = Softmax::apply<T>(scores);
        attention_weights_cache[h] = attn_weights;

        Matrix<T> head_output = attn_weights * V_h;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                output(i, q_start + j) = head_output(i, j);
            }
        }
    }

    return output * W_o;
}

template<typename T>
Matrix<T> AttentionLayer<T>::forward_prefill(const Matrix<T>& input)
{
    int seq_len = input.rows();

    Matrix<T> Q = input * W_q;
    Matrix<T> K = input * W_k;
    Matrix<T> V = input * W_v;

    if (rope_enabled) {
        applyRoPE(Q, K, 0);
        cached_pos = seq_len;
    }

    kv_K_cache = K;
    kv_V_cache = V;

    Matrix<T> output(seq_len, embed_dim);
    T scale = static_cast<T>(1.0) / static_cast<T>(std::sqrt((double)head_dim));

    for (int h = 0; h < num_heads; h++) {
        int q_start = h * head_dim;
        int kv_h = h / heads_per_group;
        int kv_start = kv_h * head_dim;

        Matrix<T> Q_h(seq_len, head_dim);
        Matrix<T> K_h(seq_len, head_dim);
        Matrix<T> V_h(seq_len, head_dim);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                Q_h(i, j) = Q(i, q_start + j);
                K_h(i, j) = K(i, kv_start + j);
                V_h(i, j) = V(i, kv_start + j);
            }
        }

        Matrix<T> scores = Q_h * K_h.transpose();

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                scores(i, j) *= scale;
            }
        }

        // Causal masking
        Masking<T> causal_mask(seq_len, seq_len);
        scores = causal_mask.apply(scores);

        Matrix<T> attn_weights = Softmax::apply<T>(scores);
        Matrix<T> head_output = attn_weights * V_h;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                output(i, q_start + j) = head_output(i, j);
            }
        }
    }

    return output * W_o;
}

template<typename T>
Matrix<T> AttentionLayer<T>::forward_cached(const Matrix<T>& input)
{
    Matrix<T> Q_new = input * W_q;
    Matrix<T> K_new = input * W_k;
    Matrix<T> V_new = input * W_v;

    if (rope_enabled) {
        applyRoPE(Q_new, K_new, cached_pos);
        cached_pos++;
    }

    kv_K_cache.appendRow(K_new);
    kv_V_cache.appendRow(V_new);

    int cached_len = kv_K_cache.rows();
    T scale = static_cast<T>(1.0) / static_cast<T>(std::sqrt((double)head_dim));

    Matrix<T> output(1, embed_dim);

    for (int h = 0; h < num_heads; h++) {
        int q_start = h * head_dim;
        int kv_h = h / heads_per_group;
        int kv_start = kv_h * head_dim;

        Matrix<T> Q_h(1, head_dim);
        Matrix<T> K_h(cached_len, head_dim);
        Matrix<T> V_h(cached_len, head_dim);

        for (int j = 0; j < head_dim; j++) {
            Q_h(0, j) = Q_new(0, q_start + j);
        }
        for (int i = 0; i < cached_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                K_h(i, j) = kv_K_cache(i, kv_start + j);
                V_h(i, j) = kv_V_cache(i, kv_start + j);
            }
        }

        // scores
        Matrix<T> scores = Q_h * K_h.transpose();
        for (int j = 0; j < cached_len; j++) {
            scores(0, j) *= scale;
        }

        // apply softmax
        Matrix<T> attn_weights = Softmax::apply<T>(scores);
        Matrix<T> head_output = attn_weights * V_h;

        for (int j = 0; j < head_dim; j++) {
            output(0, q_start + j) = head_output(0, j);
        }
    }

    return output * W_o;
}

template<typename T>
void AttentionLayer<T>::clear_cache()
{
    kv_K_cache = Matrix<T>();
    kv_V_cache = Matrix<T>();
    cached_pos = 0;
}

template<typename T>
Matrix<T> AttentionLayer<T>::backward(const Matrix<T>& grad_output)
{
    int seq_len = grad_output.rows();
    T scale = static_cast<T>(1.0) / static_cast<T>(std::sqrt((double)head_dim));

    // distribute error through output projection
    Matrix<T> grad_concat = grad_output * W_o.transpose();

    Matrix<T> concat_output(seq_len, embed_dim);
    for (int h = 0; h < num_heads; h++) {
        int q_start = h * head_dim;
        int kv_h = h / heads_per_group;
        int kv_start = kv_h * head_dim;

        Matrix<T> V_h(seq_len, head_dim);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                V_h(i, j) = V_cache(i, kv_start + j);
            }
        }
        Matrix<T> head_output = attention_weights_cache[h] * V_h;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                concat_output(i, q_start + j) = head_output(i, j);
            }
        }
    }
    grad_W_o = concat_output.transpose() * grad_output;

    Matrix<T> grad_Q(seq_len, embed_dim);
    Matrix<T> grad_K(seq_len, kv_dim);
    Matrix<T> grad_V(seq_len, kv_dim);

    for (int h = 0; h < num_heads; h++) {
        int q_start = h * head_dim;
        int kv_h = h / heads_per_group;
        int kv_start = kv_h * head_dim;

        Matrix<T> grad_head(seq_len, head_dim);
        Matrix<T> Q_h(seq_len, head_dim);
        Matrix<T> K_h(seq_len, head_dim);
        Matrix<T> V_h(seq_len, head_dim);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                grad_head(i, j) = grad_concat(i, q_start + j);
                Q_h(i, j) = Q_cache(i, q_start + j);
                K_h(i, j) = K_cache(i, kv_start + j);
                V_h(i, j) = V_cache(i, kv_start + j);
            }
        }

        Matrix<T> attn_weights = attention_weights_cache[h];

        // gradient V
        Matrix<T> grad_V_h = attn_weights.transpose() * grad_head;

        // gradient attention
        Matrix<T> grad_attn = grad_head * V_h.transpose();

        // gradient softmax
        Matrix<T> grad_scores = Softmax::derivative<T>(attn_weights, grad_attn);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                grad_scores(i, j) *= scale;
            }
        }

        // gradient Q
        Matrix<T> grad_Q_h = grad_scores * K_h;

        // gradient K
        Matrix<T> grad_K_h = grad_scores.transpose() * Q_h;

        // accumulate Q gradients
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                grad_Q(i, q_start + j) = grad_Q_h(i, j);
            }
        }

        // accumulate K/V gradients
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                grad_K(i, kv_start + j) += grad_K_h(i, j);
                grad_V(i, kv_start + j) += grad_V_h(i, j);
            }
        }
    }

    grad_W_q = input_cache.transpose() * grad_Q;
    grad_W_k = input_cache.transpose() * grad_K;
    grad_W_v = input_cache.transpose() * grad_V;

    return grad_Q * W_q.transpose() + grad_K * W_k.transpose() + grad_V * W_v.transpose();
}

template<typename T>
void AttentionLayer<T>::update(Optimizer<T>* opt)
{
    opt->step(W_q, grad_W_q);
    opt->step(W_k, grad_W_k);
    opt->step(W_v, grad_W_v);
    opt->step(W_o, grad_W_o);

    grad_W_q = Matrix<T>(embed_dim, embed_dim);
    grad_W_k = Matrix<T>(embed_dim, kv_dim);
    grad_W_v = Matrix<T>(embed_dim, kv_dim);
    grad_W_o = Matrix<T>(embed_dim, embed_dim);
}

template class AttentionLayer<float>;
template class AttentionLayer<double>;
