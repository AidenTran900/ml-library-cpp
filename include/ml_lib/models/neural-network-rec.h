#pragma once
#include "../math/matrix.h"
#include "gradient-model.h"
#include "../core/neural-network-layer.h"

class NeuralNetworkRec : public GradientModel<> {
    private:
        std::vector<NeuralNetworkLayer<>> layers;

        Matrix<> last_input;
        Matrix<> last_output;

    public:
        NeuralNetworkRec(int input_dim, std::unique_ptr<LossFunction<>> loss, std::unique_ptr<Optimizer<>> opt, std::unique_ptr<Regularizer<>> reg);

        void addLayer(int input_dim, int output_dim, ACTIVATION_FUNC act);

        Matrix<> forward(const Matrix<>& X) override;
        void backward(const Matrix<>& y_true) override;
        void update() override;
};
