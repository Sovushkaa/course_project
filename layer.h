#pragma once

#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace Eigen;

void ReadMatrix(std::ifstream &ss, MatrixXd &A) {
    size_t rows, cols;
    ss >> rows >> cols;
    A.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ss >> A(i, j);
        }
    }
}

void WriteMatrix(std::ofstream &ss, const MatrixXd &A) {
    ss << A.rows() << " " << A.cols() << std::endl;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            ss << A(i, j) << " ";
        }
        ss << std::endl;
    }
}

auto sigmoid0 = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
auto sigmoid1 = [](double x) {
    return std::exp(-x) / ((std::exp(-x) + 1) * (std::exp(-x) + 1));
};
// надо будет поменять сигмоиду на что-то ещё

class Layer {
public:
    Layer();
    Layer (size_t input_size, size_t output_size, std::function<double(double)> f0 = sigmoid0,
     std::function<double(double)> f1 = sigmoid1);
    Layer(const MatrixXd& A, const MatrixXd& b, std::function<double(double)> f0 = sigmoid0,
     std::function<double(double)> f1 = sigmoid1);
    
    MatrixXd Evaluate(const MatrixXd& x);
    MatrixXd Gradient_A(const MatrixXd& x, const MatrixXd& u);
    MatrixXd Gradient_b(const MatrixXd& x, const MatrixXd& u);
    MatrixXd Push(const MatrixXd& x, const MatrixXd& u);
    void Update(double coeff, const MatrixXd& x, const MatrixXd& initial_line);
    void Write_A(std::ifstream &ss);
    void Write_b(std::ifstream &ss);
    
private:
    MatrixXd A_;
    MatrixXd b_;
    std::function<double(double)> f0_;
    std::function<double(double)> f1_;
    MatrixXd x; 
    MatrixXd Diag_matrix(const MatrixXd& x, const MatrixXd& u);
};