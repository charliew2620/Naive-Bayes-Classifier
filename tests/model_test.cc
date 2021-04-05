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

TEST_CASE("Tests pixel probabilities") {
    std::ifstream input("../../../data/testingfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;
    naivebayes::Model model(training_data);
    
    

    SECTION("Checks row size") {
        REQUIRE(model.GetPixelProbabilities().size() == 5);
    }
    
    SECTION("Checks column size") {
        REQUIRE(model.GetPixelProbabilities()[0].size() == 5);
    }
    
    SECTION("Checks label size") {
        REQUIRE(model.GetPixelProbabilities()[0][0].size() == 10);
    }
    
    SECTION("Checks shade size") {
        REQUIRE(model.GetPixelProbabilities()[0][0][0].size() == 3);
    }
}

