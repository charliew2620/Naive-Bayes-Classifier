#include <catch2/catch.hpp>

#include <core/load_data.h>
#include <iostream>
#include <fstream>
#include <core/model.h>


TEST_CASE("Tests accuracy") {
std::ifstream stream("../../../data/trainingimagesandlabels.txt");
naivebayes::TrainingData training_data(28, 5000);
stream >> training_data;

naivebayes::Model model(training_data);

std::ofstream os;
os.open("../../../data/wheredidmyweekendgo.txt");
os << model;


std::ifstream stream1("../../../data/testimagesandlabels.txt");
naivebayes::TrainingData testing_data(28, 1000);
stream1 >> testing_data;

REQUIRE(model.ComputeAccuracy(testing_data.GetImages()) == 0.772);
os.close();
}