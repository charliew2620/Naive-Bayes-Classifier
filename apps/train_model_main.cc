#include <iostream>

#include <core/training_data.h>
#include <fstream>
#include <core/model.h>


int main(int argc, char** argv) {
    if (argc >= 2) {
        std::ifstream stream(argv[1]);
        naivebayes::TrainingData training_data(28, 5000);
        stream >> training_data;
        
        std::cout << "lmao1" << std::endl;
        
        naivebayes::Model model(training_data);

        std::cout << "lmao0" << std::endl;
        
        std::ofstream os;
        os.open("../data/lmao.txt");
        os << model;
        os.close();
        
        return 0;
    }
  return 1;
}
