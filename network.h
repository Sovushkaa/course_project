#pragma once

#include "layer.h"
#include "penalty_function.h"
#include <vector>

class Network {
public:
    Network(size_t input_size, size_t output_size, size_t layers_number);
    Network(const std::string& filename);
    
    MatrixXd Evaluate(MatrixXd& x);
    void Iteration(MatrixXd& x, MatrixXd& y);
    void SaveToFile(const std::string& filename);

private:
    size_t layers_number_;
    std::vector<Layer> layers_;
    PenaltyFunction penalty_func_;
};