#include <iostream>

#include <core/training_data.h>
#include <fstream>
#include <core/model.h>


int main(int argc, char** argv) {
    if (argc >= 2) {
        std::ifstream stream(argv[1]);
        naivebayes::TrainingData training_data(28, 5000);
        stream >> training_data;
        
        naivebayes::Model model(training_data);
        
        std::ofstream os;
        os.open("../data/lmao.txt");
        os << model;
        os.close();
        
        std::ifstream ifstream("../data/lmao.txt");
        naivebayes::Model model1(28);
        
        return 0;
    }
  return 1;
}
