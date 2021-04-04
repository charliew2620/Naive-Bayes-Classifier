#include <iostream>

#include <core/training_data.h>
#include <fstream>

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::ifstream stream(argv[1]);
        naivebayes::TrainingData training_data(28, 5000);
        stream >> training_data;

        return 0;
    }
  return 1;
}
