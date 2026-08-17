#include "ml_lib/utils/llama-loader.h"
#include "ml_lib/core/token-sampler.h"
#include <functional>

int main() {
    Tokenizer tokenizer;
    auto model = LlamaLoader<float>::load("examples/datasets/language-model/SmolLM2-135M-Instruct-Q8_0.gguf", tokenizer, 2048);

    // Ensure chat stop tokens are registered (GGUF may only provide eos_token_id)
    for (const auto& tok : {"<|eot_id|>", "<|end_of_text|>"}) {
        int id = tokenizer.tokenToId(tok);
        if (id >= 0) model->addStopToken(id);
    }

#ifdef ML_USE_CUDA
    model->prepare_gpu();
#endif

    TokenSampler<float> sampler;
    sampler.addProcessor(std::make_unique<TemperatureProcessor<float>>(0.7));
    sampler.addProcessor(std::make_unique<TopPProcessor<float>>(0.9));
    sampler.setSelector(std::make_unique<CategoricalSelector<float>>());

    auto stream = [&](int token) {
        std::cout << tokenizer.decode({token}) << std::flush;
    };

    std::string input;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, input) || input == "exit") break;
        if (input.empty()) continue;

       std::string formatted =
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant.<|eot_id|>\n"

            "<|start_header_id|>user<|end_header_id|>\n"
            + input +
            "\n<|eot_id|>\n"
            
            "<|start_header_id|>assistant<|end_header_id|>";
        std::vector<int> prompt = tokenizer.encode(formatted);
#ifdef ML_USE_CUDA
        model->generate_gpu(prompt, 2048, sampler, stream);
#else
        model->generate(prompt, 2048, sampler, stream);
#endif
        std::cout << std::endl;
    }

    return 0;
}
