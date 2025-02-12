
#include "network.h"
#include "penalty_function.h"
#include <fstream>

Network::Network(size_t input_size, size_t output_size, size_t layers_number)
    : layers_number_(layers_number) {
    Layer head = Layer(input_size, output_size);
    layers_.push_back(head);
    for (size_t i = 1; i < layers_number; ++i) {
        Layer cur_layer = Layer(output_size, output_size);
        layers_.push_back(cur_layer);
    }
}

Network::Network(const std::string& filename) {
    std::ifstream ss;
    ss.open(filename);
    ss >> layers_number_;
    for (size_t index = 0; index < layers_number_; ++index) {
        MatrixXd A, b;
        ReadMatrix(ss, A);
        ReadMatrix(ss, b);
        Layer cur_layer = Layer(A, b);
        layers_.push_back(cur_layer);
    }
    ss.close();
}

MatrixXd Network::Evaluate(MatrixXd& x) {
    MatrixXd result = layers_[0].Evaluate(x);
    for (size_t i = 1; i < layers_number_; ++i) {
        result = layers_[i].Evaluate(result);
    }
    return result;
}

void Network::Iteration(MatrixXd& x, MatrixXd& y) {
    MatrixXd z = Evaluate(x);
    MatrixXd cur_line = penalty_func_.InitialLine(z, y);
    double coeff = 0.01;
    for (int i = layers_number_ - 1; i >= 0; --i) {
        layers_[i].Update(coeff, x, cur_line);
        cur_line = layers_[i].Push(x, cur_line);
    }
}

void Network::SaveToFile(const std::string& filename) {
    std::ofstream ss;
    ss.open(filename);
    ss << layers_number_ << std::endl;
    for (auto layer : layers_) {
        layer.write_A();
        layer.write_b();
        //WriteMatrix(ss, layer.A_);
        //WriteMatrix(ss, layer.b_);
    }
    ss.close();
}