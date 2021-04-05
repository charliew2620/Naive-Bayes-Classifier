#include <catch2/catch.hpp>

#include <core/training_data.h>
#include <core/model.h>
#include <iostream>
#include <fstream>

TEST_CASE("Tests class probabilities") {
    std::ifstream input("../../../data/testingfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;
    naivebayes::Model model(training_data);

    SECTION("Checks size") {
        REQUIRE(model.GetClassProbabilities().size() == 10);
    }
    
    SECTION("Checks probability for class 0") {
        REQUIRE(model.GetClassProbabilities()[0] == Approx(0.1538).epsilon(0.01));
    }

    SECTION("Checks probability for class 1") {
        REQUIRE(model.GetClassProbabilities()[1] == Approx(0.1538).epsilon(0.01));
    }

    SECTION("Checks probability for class 2") {
        REQUIRE(model.GetClassProbabilities()[2] == Approx(0.0769).epsilon(0.01));
    }

    SECTION("Checks probability for class 7") {
        REQUIRE(model.GetClassProbabilities()[7] == Approx(0.1538).epsilon(0.01));
    }
}

