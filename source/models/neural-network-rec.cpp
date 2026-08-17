#include "neural-network-ff.h"
#include "neural-network-layer.h"
#include <algorithm>

NeuralNetworkRecurrent::NeuralNetworkRecurrent(int input_dim, std::unique_ptr<LossFunction<>> loss, std::unique_ptr<Optimizer<>> opt, std::unique_ptr<Regularizer<>> reg)
    : GradientModel<>(std::move(loss), std::move(opt), std::move(reg))
{}

void NeuralNetworkRecurrent::addLayer(int input_dim, int output_dim, ACTIVATION_FUNC act)
{
    this->layers.push_back(NeuralNetworkLayer<>(input_dim, output_dim, act));
}

Matrix<> NeuralNetworkRecurrent::forward(const Matrix<> &X)
{
    last_input = X;
    for (auto& layer : layers) {
        last_input = layer.forward(last_input);
    }
    last_output = last_input;

    return last_output;
}

void NeuralNetworkRecurrent::backward(const Matrix<> &y_true)
{
    Matrix<> last_gradient = loss_func->gradient(last_output, y_true);
    for (int i = layers.size() - 1; i >= 0; i--) {
        last_gradient = layers[i].backward(last_gradient);
    }
}

void NeuralNetworkRecurrent::update()
{
    for (auto& layer : layers) {
        layer.update(optimizer.get());
    }
}
