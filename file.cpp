#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace Eigen;

template<typename OutputType, typename... InputTypes>
class FunctionPro {
public:
    using FunctionType = std::function<OutputType(InputTypes...)>;

    FunctionPro(FunctionType function, FunctionType derivative)
        : function_(function), derivative_(derivative) {
    }

    OutputType Evaluate0(InputTypes&... x) {
        return function_(x...);
    }

    OutputType Evaluate1(InputTypes&... x) {
        return derivative_(x...);
    }

private:
    FunctionType function_;
    FunctionType derivative_;
};

void PrintSize(MatrixXd& A) {
    std::cout << A.rows() << " " << A.cols() << " " << std::endl;
}

void ReadMatrix(std::ifstream& ss, MatrixXd& A) {
    size_t rows, cols;
    ss >> rows >> cols;
    A.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ss >> A(i, j);
        }
    }
}

void WriteMatrix(std::ofstream& ss, const MatrixXd& A) {
    ss << A.rows() << " " << A.cols() << std::endl;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            ss << A(i, j) << " ";
        }
        ss << std::endl;
    }
}


auto sigmoid0 = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
auto sigmoid1 = [](double x) { return std::exp(-x) / ((std::exp(-x) + 1) * (std::exp(-x) + 1)); };
FunctionPro<double, double> sigmoid(sigmoid0, sigmoid1);
//надо будет поменять сигмоиду на что-то ещё

class Layer {
public:
    Layer() : sigma_(sigmoid) {
    } //без этого у меня код не компилировался, я просто добавила пока

    Layer(size_t input_size, size_t output_size, FunctionPro<double, double>& sigma = sigmoid) 
        : A_(MatrixXd::Random(output_size, input_size)), 
          b_(MatrixXd::Random(output_size, 1)),
          sigma_(sigma) {
    }

    Layer(const MatrixXd& A, const MatrixXd& b, FunctionPro<double, double>& sigma = sigmoid)
     : A_(A), b_(b), sigma_(sigma) {
    }

    MatrixXd Evaluate(const MatrixXd& x) {
        MatrixXd after_linear = A_ * x + b_;
        return after_linear.unaryExpr([this](double val) { return sigma_.Evaluate0(val); }); //пока оно поэлементное
    }

    MatrixXd Gradient_A (const MatrixXd& x, const MatrixXd& u) {
        return Diag_matrix(x, u) * u.transpose() * x.transpose();
    }

    MatrixXd Gradient_b (const MatrixXd& x, const MatrixXd& u) {
        return Diag_matrix(x, u) * u.transpose();
    }

    MatrixXd Push (const MatrixXd& x, const MatrixXd& u) {
        return u * Diag_matrix(x, u) * A_;
    }

    void Update (double coeff, const MatrixXd& x, const MatrixXd& initial_line) {
        MatrixXd gradient_A = Gradient_A(x, initial_line);
        MatrixXd gradient_b = Gradient_b(x, initial_line);
        A_ -= gradient_A * coeff;
        b_ -= gradient_b * coeff;
    }

private:
    MatrixXd A_;
    MatrixXd b_;
    FunctionPro<double, double> sigma_;
    MatrixXd x; //вход, который пришёл блоку, надо будет в какой-то момент сохранять
    friend class Network;

    MatrixXd Diag_matrix (const MatrixXd& x, const MatrixXd& u) {
        MatrixXd after_linear = A_ * x + b_;
        MatrixXd diag_values = 
            after_linear.unaryExpr([this](double val) { return sigma_.Evaluate1(val); });
        return diag_values.asDiagonal();
    }

};

class PenaltyFunction{
public:

    PenaltyFunction() {
    }

    double Evaluate(const MatrixXd& z, const MatrixXd& y) {
        return func0_(z, y);
    }

    MatrixXd InitialLine (const MatrixXd& z, const MatrixXd& y) {
        return func1_(z, y).transpose();
    }

    double AverageMistake(const MatrixXd& z, const MatrixXd& y) {
        double total_error = 0.0;
        for (size_t i = 0; i < z.cols(); ++i) {
            total_error += Evaluate(z.col(i), y.col(i));
        }
        return total_error / z.cols();
    }

private:
    std::function<double(const MatrixXd&, const MatrixXd&)> func0_ = 
        [](const MatrixXd& z, const MatrixXd& y) { return (z - y).squaredNorm(); };
    std::function<MatrixXd(const MatrixXd&, const MatrixXd&)> func1_ = 
        [](const MatrixXd& z, const MatrixXd& y) { return 2 * (z - y); };
    //через FunctionPro стало сложно писать
};

class Network {
public:
    Network(size_t input_size, size_t output_size, size_t layers_number) : 
     layers_number_(layers_number) {
        //тут надо будет поменять размеры промежуточных слоёв, так слишком большое получается
        Layer head = Layer(input_size, output_size);
        layers_.push_back(head);
        for (size_t i = 1; i < layers_number; ++i) {
            Layer cur_layer = Layer(output_size, output_size);
            layers_.push_back(cur_layer);
        }
    }

    Network(const std::string& filename) {
        std::ifstream ss;
        ss.open(filename);
        ss >> layers_number_;
        for(size_t index = 0; index < layers_number_; ++index) {
            MatrixXd A, b;
            ReadMatrix(ss, A);
            ReadMatrix(ss, b);
            Layer cur_layer = Layer(A, b);
            layers_.push_back(cur_layer);
        }
        ss.close();
    }

    MatrixXd Evaluate(MatrixXd& x) {
        MatrixXd result = layers_[0].Evaluate(x);
        for (size_t i = 1; i < layers_number_; ++i) {
            result = layers_[i].Evaluate(result);
        }
        return result;
    }

    //тут пока что фигня написана, надо x менять и сохранять внутри блоков, а у меня везде один
    void Iteration(MatrixXd& x, MatrixXd& y) {
        MatrixXd z = Evaluate(x);
        MatrixXd cur_line = penalty_func_.InitialLine (z, y);
        double coeff = 0.01;
        for (int i = layers_number_ - 1; i >= 0; --i) {
            layers_[i].Update(coeff, x, cur_line);
            cur_line = layers_[i].Push(x, cur_line);
        }
    }

    void SaveToFile(const std::string& filename) {
        std::ofstream ss;
        ss.open(filename);
        ss << layers_number_ << std::endl;
        for (auto layer: layers_) {
            WriteMatrix(ss, layer.A_);
            WriteMatrix(ss, layer.b_);
        }
        ss.close();
    }

private:
    size_t layers_number_;
    std::vector<Layer> layers_;
    PenaltyFunction penalty_func_ = PenaltyFunction();
};

struct Number {
    MatrixXd label; // 1 in correct value, 0 in others
    MatrixXd pixels; //input_size = 784
    int correct_value;

    Number() : label(10, 1), pixels(784, 1) {
        label.setZero();
        pixels.setZero();
    }
};


std::vector<Number> ReadCSV(const std::string& filename) {
    std::vector<Number> result;
    std::ifstream file(filename);
    std::string cur_line, cur_label, cur_pixel;
    std::getline(file, cur_line);
    while (std::getline(file, cur_line)) {
        std::stringstream ss(cur_line);
        Number number = Number();
        std::getline(ss, cur_label, ',');
        number.correct_value = std::stoi(cur_label);
        number.label(number.correct_value, 0) = 1;
        int cur_index = 0;
        while (std::getline(ss, cur_pixel, ',')) {
            number.pixels(cur_index, 0) = std::stod(cur_pixel);
            ++cur_index;
        }
        result.push_back(number);
    }
    file.close();
    return result;
}


int main() {

    //это всё надо кнчн не в мэйне делать

    std::vector<Number> train_data = ReadCSV("mnist_train.csv");
    std::vector<Number> test_data = ReadCSV("mnist_test.csv");

    std::cout << train_data.size() << std::endl; //60.000
    std::cout << test_data.size() << std::endl; //10.000

    //Network network("random_network.txt");
    //network.SaveToFile("random_network.txt");

    Network network = Network(784, 10, 3);


    //тут я пыталась потестить как-то нейросеть с неверными формулами
    //ещё и на батчи не разбила, по одному элементу гоняла
    //крч с этого момента начинается полный ужас мрак
    int it = 0;
    for (auto el: train_data) {
        network.Iteration(el.pixels, el.label);
        ++it;
        if (it % 10 == 0) {
            std::cout << it << std::endl;
        }
        if (it > 500) {
            break;
        }
    }
    double correct_numbers = 0;
    PenaltyFunction penalty_func_ = PenaltyFunction();
    it = 0;
    for (auto el : test_data) {
        MatrixXd z = network.Evaluate(el.pixels);
        int result = 0;
        for (int i = 1; i < 10; ++i) {
            if (abs(z(i, 0)) > abs(z(result, 0))) {
                result = i;
            } 
        }
        if (result == el.correct_value) {
            ++correct_numbers;
        }
        ++it;
        if (it > 1000) {
            break;
        }
    }
    std::cout << "Correct numbers: " << correct_numbers << std::endl;
    std::cout << "Wrong numbers: " << train_data.size() - correct_numbers << std::endl;

    return 0;
}