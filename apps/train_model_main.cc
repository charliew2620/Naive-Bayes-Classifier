#include <iostream>

#include <core/load_data.h>
#include <fstream>
#include <core/model.h>

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::ifstream stream(argv[1]);
        naivebayes::LoadData training_data(28, 5000);
        stream >> training_data;

        naivebayes::Model model(training_data);

        std::ofstream os;
        os.open("../data/wheredidmyweekendgo.txt");
        os << model;


        std::ifstream stream1(argv[2]);
        naivebayes::LoadData testing_data(28, 1000);
        stream1 >> testing_data;
        
        std::cout << model.ComputeAccuracy(testing_data.GetImages());
        os.close();
        return 0;
    }
  return 1;
}
