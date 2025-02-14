#pragma once

#include <Eigen/Dense>
#include <functional>

class PenaltyFunction {
public:
    PenaltyFunction();
    
    double Evaluate(const Eigen::MatrixXd& z, const Eigen::MatrixXd& y);
    Eigen::MatrixXd InitialLine(const Eigen::MatrixXd& z, const Eigen::MatrixXd& y);
    double AverageMistake(const Eigen::MatrixXd& z, const Eigen::MatrixXd& y);
    
private:
    std::function<double(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> func0_;
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&)> func1_;
};