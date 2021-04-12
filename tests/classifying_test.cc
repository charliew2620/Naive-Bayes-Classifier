#include <catch2/catch.hpp>

#include <core/load_data.h>
#include <iostream>
#include <fstream>
#include <core/model.h>


TEST_CASE("Tests accuracy for given test file") {
std::ifstream stream("../../../data/trainingimagesandlabels.txt");
naivebayes::LoadData training_data(28, 5000);
stream >> training_data;

naivebayes::Model model(training_data);

std::ofstream os;
os.open("../../../data/wheredidmyweekendgo.txt");
os << model;


std::ifstream stream1("../../../data/testimagesandlabels.txt");
naivebayes::LoadData testing_data(28, 1000);
stream1 >> testing_data;

REQUIRE(model.ComputeAccuracy(testing_data.GetImages()) == 0.772);
os.close();
}

TEST_CASE("Tests accuracy") {
    std::ifstream stream("../../../data/trainingimagesandlabels.txt");
    naivebayes::LoadData training_data(28, 5000);
    stream >> training_data;

    naivebayes::Model model(training_data);

    std::ofstream os;
    os.open("../../../data/wheredidmyweekendgo.txt");
    os << model;


    std::ifstream stream1("../../../data/28by28classifytestfile.txt");
    naivebayes::LoadData testing_data(28, 3);
    stream1 >> testing_data;

    REQUIRE(model.ComputeAccuracy(testing_data.GetImages()) == 1);
    os.close();
}