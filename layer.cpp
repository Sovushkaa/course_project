#include "layer.h"

Layer::Layer()
    : f0_(sigmoid0), f1_(sigmoid1) {}

Layer::Layer(size_t input_size, size_t output_size, std::function<double(double)> f0, std::function<double(double)> f1)
    : A_(MatrixXd::Random(output_size, input_size)), b_(MatrixXd::Random(output_size, 1)), f0_(f0), f1_(f1) {}

Layer::Layer(const MatrixXd& A, const MatrixXd& b, std::function<double(double)> f0, std::function<double(double)> f1)
    : A_(A), b_(b), f0_(f0), f1_(f1) {}

MatrixXd Layer::Evaluate(const MatrixXd& x) {
    MatrixXd after_linear = A_ * x + b_;
    return after_linear.unaryExpr([this](double val) { return f0_(val); });
}

MatrixXd Layer::Gradient_A(const MatrixXd& x, const MatrixXd& u) {
    return Diag_matrix(x, u) * u.transpose() * x.transpose();
}

MatrixXd Layer::Gradient_b(const MatrixXd& x, const MatrixXd& u) {
    return Diag_matrix(x, u) * u.transpose();
}

MatrixXd Layer::Push(const MatrixXd& x, const MatrixXd& u) {
    return u * Diag_matrix(x, u) * A_;
}

void Layer::Update(double coeff, const MatrixXd& x, const MatrixXd& initial_line) {
    MatrixXd gradient_A = Gradient_A(x, initial_line);
    MatrixXd gradient_b = Gradient_b(x, initial_line);
    A_ -= gradient_A * coeff;
    b_ -= gradient_b * coeff;
}

MatrixXd Layer::Diag_matrix(const MatrixXd& x, const MatrixXd& u) {
    MatrixXd after_linear = A_ * x + b_;
    MatrixXd diag_values = after_linear.unaryExpr([this](double val) { return f1_(val); });
    return diag_values.asDiagonal();
}

void Layer::Write_A(std::ifstream &ss) {
    WriteMatrix(ss, A);
}

void Layer::Write_b(std::ifstream &ss) {
    WriteMatrix(ss, b);
}