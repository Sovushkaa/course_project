
#include "penalty_function.h"

PenaltyFunction::PenaltyFunction() {
    func0_ = [](const Eigen::MatrixXd& z, const Eigen::MatrixXd& y) {
        return (z - y).squaredNorm();
    };
    func1_ = [](const Eigen::MatrixXd& z, const Eigen::MatrixXd& y) {
        return 2 * (z - y);
    };
}

double PenaltyFunction::Evaluate(const Eigen::MatrixXd& z, const Eigen::MatrixXd& y) {
    return func0_(z, y);
}

Eigen::MatrixXd PenaltyFunction::InitialLine(const Eigen::MatrixXd& z, const Eigen::MatrixXd& y) {
    return func1_(z, y).transpose();
}

double PenaltyFunction::AverageMistake(const Eigen::MatrixXd& z, const Eigen::MatrixXd& y) {
    double total_error = 0.0;
    for (size_t i = 0; i < z.cols(); ++i) {
        total_error += Evaluate(z.col(i), y.col(i));
    }
    return total_error / z.cols();
}