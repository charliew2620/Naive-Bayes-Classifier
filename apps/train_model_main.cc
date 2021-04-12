#include <iostream>

#include <core/load_data.h>
#include <fstream>
#include <core/model.h>
#include <core/classifier.h>

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::ifstream stream(argv[1]);
        naivebayes::TrainingData training_data(28, 5000);
        stream >> training_data;

        naivebayes::Model model(training_data);

        std::ofstream os;
        os.open("../data/wheredidmyweekendgo.txt");
        os << model;
        os.close();
        return 0;
    }
  return 1;
}
