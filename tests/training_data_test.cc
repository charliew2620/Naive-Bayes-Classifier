#include <catch2/catch.hpp>

#include <core/training_data.h>
#include <iostream>
#include <fstream>

TEST_CASE("Tests getters") {
    std::ifstream input("../../../data/testingfile.txt");
    naivebayes::TrainingData training_data(3, 3);
    input >> training_data;

    SECTION("Tests getImageCount") {
        REQUIRE(training_data.GetImageCount() == 3);
    }
    
    SECTION("Tests getImageSize") {
        REQUIRE(training_data.GetImageSize() == 3);
    }

    SECTION("Tests getImages") {
        
    }

}