#include <iostream>

#include <core/training_data.h>
#include <fstream>

// TODO: You may want to change main's signature to take in argc and argv
int main(int arg, char** argv) {

    if (arg >= 2) {
        std::ifstream stream(argv[1]);
        naivebayes::TrainingData training_data(28, 5000);
        stream >> training_data;

        return 0;
    }
  return 1;
}
