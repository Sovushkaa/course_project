#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "network.h"
#include "penalty_function.h"

using namespace Eigen;

void PrintSize(MatrixXd &A) {
    std::cout << A.rows() << " " << A.cols() << " " << std::endl;
}

struct Number {
    MatrixXd label;  // 1 in correct value, 0 in others
    MatrixXd pixels; // input_size = 784
    int correct_value;

    Number() : label(10, 1), pixels(784, 1) {
        label.setZero();
        pixels.setZero();
    }
};

std::vector<Number> ReadCSV(const std::string &filename) {
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

    // это всё надо кнчн не в мэйне делать

    std::vector<Number> train_data = ReadCSV("mnist_train.csv");
    std::vector<Number> test_data = ReadCSV("mnist_test.csv");

    std::cout << train_data.size() << std::endl; // 60.000
    std::cout << test_data.size() << std::endl;  // 10.000

    // Network network("random_network.txt");
    // network.SaveToFile("random_network.txt");

    Network network = Network(784, 10, 3);

    // тут я пыталась потестить как-то нейросеть с неверными формулами
    // ещё и на батчи не разбила, по одному элементу гоняла
    // крч с этого момента начинается полный ужас мрак
    int it = 0;
    for (auto el : train_data) {
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
    std::cout << "Wrong numbers: " << train_data.size() - correct_numbers
              << std::endl;

    return 0;
}