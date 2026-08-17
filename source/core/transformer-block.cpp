#include "ml_lib/core/transformer-block.h"
#include <cmath>

// Legacy constructor — preserves original behavior
template<typename T>
TransformerBlock<T>::TransformerBlock(int embed_dim, int num_heads, int ff_dim)
    : TransformerBlock(embed_dim, num_heads, ff_dim,
                       NormType::LAYER_NORM, FFNType::STANDARD, NormPosition::POST_NORM)
{}

// Configurable constructor
template<typename T>
TransformerBlock<T>::TransformerBlock(int embed_dim, int num_heads, int ff_dim,
                                     NormType norm_type, FFNType ffn_type,
                                     NormPosition norm_position,
                                     int num_kv_heads)
    : norm_type(norm_type), ffn_type(ffn_type), norm_position(norm_position),
      attention(embed_dim, num_heads, num_kv_heads < 0 ? num_heads : num_kv_heads),
      ff1(embed_dim, ff_dim,
          ffn_type == FFNType::STANDARD ? ACTIVATION_FUNC::RELU : ACTIVATION_FUNC::LINEAR),
      ff2(ff_dim, embed_dim, ACTIVATION_FUNC::LINEAR)
{
    if (norm_type == NormType::LAYER_NORM) {
        ln1 = std::make_unique<LayerNorm<T>>(embed_dim);
        ln2 = std::make_unique<LayerNorm<T>>(embed_dim);
    } else {
        rms1 = std::make_unique<RMSNorm<T>>(embed_dim);
        rms2 = std::make_unique<RMSNorm<T>>(embed_dim);
    }

    if (ffn_type == FFNType::GATED_SILU) {
        ff_gate = std::make_unique<NeuralNetworkLayer<T>>(embed_dim, ff_dim, ACTIVATION_FUNC::LINEAR);
    }
}

// Norm dispatch helpers

template<typename T>
Matrix<T> TransformerBlock<T>::normForward1(const Matrix<T>& x) {
    return (norm_type == NormType::LAYER_NORM) ? ln1->forward(x) : rms1->forward(x);
}

template<typename T>
Matrix<T> TransformerBlock<T>::normForward2(const Matrix<T>& x) {
    return (norm_type == NormType::LAYER_NORM) ? ln2->forward(x) : rms2->forward(x);
}

template<typename T>
Matrix<T> TransformerBlock<T>::normBackward1(const Matrix<T>& g) {
    return (norm_type == NormType::LAYER_NORM) ? ln1->backward(g) : rms1->backward(g);
}

template<typename T>
Matrix<T> TransformerBlock<T>::normBackward2(const Matrix<T>& g) {
    return (norm_type == NormType::LAYER_NORM) ? ln2->backward(g) : rms2->backward(g);
}

// FFN dispatch

template<typename T>
Matrix<T> TransformerBlock<T>::ffnForward(const Matrix<T>& x) {
    if (ffn_type == FFNType::STANDARD) {
        return ff2.forward(ff1.forward(x));
    }

    // gated SiLU
    Matrix<T> gate_out = ff_gate->forward(x);
    Matrix<T> up_out = ff1.forward(x);

    // apply SiLU to gate
    for (int i = 0; i < gate_out.rows(); i++) {
        for (int j = 0; j < gate_out.cols(); j++) {
            double val = static_cast<double>(gate_out(i, j));
            gate_out(i, j) = static_cast<T>(val / (1.0 + std::exp(-val)));
        }
    }

    gate_cache = gate_out;
    up_cache = up_out;

    return ff2.forward(gate_out.hadamard(up_out));
}

template<typename T>
Matrix<T> TransformerBlock<T>::ffnBackward(const Matrix<T>& g) {
    if (ffn_type == FFNType::STANDARD) {
        return ff1.backward(ff2.backward(g));
    }

    // gated SiLU backward
    Matrix<T> grad_hidden = ff2.backward(g);

    Matrix<T> grad_gate_act = grad_hidden.hadamard(up_cache);
    Matrix<T> grad_up = grad_hidden.hadamard(gate_cache);

    Matrix<T> grad_ff1 = ff1.backward(grad_up);

    // SiLU derivative
    Matrix<T> grad_gate = ff_gate->backward(grad_gate_act);

    return grad_ff1 + grad_gate;
}

// Forward

template<typename T>
Matrix<T> TransformerBlock<T>::forward(const Matrix<T>& input)
{
    attention_input_cache = input;

    if (norm_position == NormPosition::PRE_NORM) {
        // Pre-norm
        Matrix<T> normed1 = normForward1(input);
        Matrix<T> attn_out = attention.forward(normed1);
        Matrix<T> residual1 = input + attn_out;

        ff_input_cache = residual1;
        Matrix<T> normed2 = normForward2(residual1);
        Matrix<T> ff_out = ffnForward(normed2);
        return residual1 + ff_out;
    }

    // Post-norm
    Matrix<T> attn_out = attention.forward(input);
    Matrix<T> residual1 = input + attn_out;
    Matrix<T> normed1 = normForward1(residual1);

    ff_input_cache = normed1;
    Matrix<T> ff_out = ffnForward(normed1);
    Matrix<T> residual2 = normed1 + ff_out;
    return normForward2(residual2);
}

template<typename T>
Matrix<T> TransformerBlock<T>::forward_cached(const Matrix<T>& input)
{
    if (norm_position == NormPosition::PRE_NORM) {
        Matrix<T> normed1 = normForward1(input);
        Matrix<T> attn_out = attention.forward_cached(normed1);
        Matrix<T> residual1 = input + attn_out;

        Matrix<T> normed2 = normForward2(residual1);
        Matrix<T> ff_out = ffnForward(normed2);
        return residual1 + ff_out;
    }

    Matrix<T> attn_out = attention.forward_cached(input);
    Matrix<T> residual1 = input + attn_out;
    Matrix<T> normed1 = normForward1(residual1);

    Matrix<T> ff_out = ffnForward(normed1);
    Matrix<T> residual2 = normed1 + ff_out;
    return normForward2(residual2);
}

template<typename T>
Matrix<T> TransformerBlock<T>::forward_prefill(const Matrix<T>& input)
{
    if (norm_position == NormPosition::PRE_NORM) {
        Matrix<T> normed1 = normForward1(input);
        Matrix<T> attn_out = attention.forward_prefill(normed1);
        Matrix<T> residual1 = input + attn_out;

        Matrix<T> normed2 = normForward2(residual1);
        Matrix<T> ff_out = ffnForward(normed2);
        return residual1 + ff_out;
    }

    Matrix<T> attn_out = attention.forward_prefill(input);
    Matrix<T> residual1 = input + attn_out;
    Matrix<T> normed1 = normForward1(residual1);

    Matrix<T> ff_out = ffnForward(normed1);
    Matrix<T> residual2 = normed1 + ff_out;
    return normForward2(residual2);
}

template<typename T>
void TransformerBlock<T>::clear_cache()
{
    attention.clear_cache();
}

// Backward

template<typename T>
Matrix<T> TransformerBlock<T>::backward(const Matrix<T>& grad_output)
{
    if (norm_position == NormPosition::PRE_NORM) {
        // reverse of pre-norm forward
        Matrix<T> grad_ff = ffnBackward(normBackward2(grad_output));
        Matrix<T> grad_residual1 = grad_output + grad_ff;
        Matrix<T> grad_attn = attention.backward(normBackward1(grad_residual1));
        return grad_residual1 + grad_attn;
    }

    // reverse of post-norm forward
    Matrix<T> grad_norm2 = normBackward2(grad_output);
    Matrix<T> grad_ff = ffnBackward(grad_norm2);
    Matrix<T> grad_residual1 = grad_norm2 + grad_ff;
    Matrix<T> grad_norm1 = normBackward1(grad_residual1);
    Matrix<T> grad_attn = attention.backward(grad_norm1);
    return grad_norm1 + grad_attn;
}

template<typename T>
void TransformerBlock<T>::update(Optimizer<T>* opt)
{
    attention.update(opt);

    if (norm_type == NormType::LAYER_NORM) {
        ln1->update(opt);
        ln2->update(opt);
    } else {
        rms1->update(opt);
        rms2->update(opt);
    }

    ff1.update(opt);
    ff2.update(opt);
    if (ff_gate) ff_gate->update(opt);
}

template class TransformerBlock<float>;
template class TransformerBlock<double>;
