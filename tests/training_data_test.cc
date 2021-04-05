#include <catch2/catch.hpp>

#include <core/training_data.h>
#include <iostream>
#include <fstream>

TEST_CASE("Tests basic getters") {
    std::ifstream input("../../../data/testingfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;

    SECTION("Tests getImageCount") {
        REQUIRE(training_data.GetImageCount() == 3);
    }
    
    SECTION("Tests getImageSize") {
        REQUIRE(training_data.GetImageSize() == 5);
    }
}

TEST_CASE("Tests GetImages") {
    std::ifstream input("../../../data/testingfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;
    SECTION("First image") {
        std::vector<std::vector<int>> pixels
                {
                        {2, 2, 2, 1, 1},
                        {1, 0, 0, 0, 2},
                        {1, 0, 0, 0, 2},
                        {1, 0, 0, 0, 2},
                        {2, 2, 2, 2, 1}
                };
        naivebayes::Image image(0, 5, pixels);
        REQUIRE(training_data.GetImages()[0].GetLabel() == image.GetLabel());
        REQUIRE(training_data.GetImages()[0].GetImageSize() == image.GetImageSize());
        REQUIRE(training_data.GetImages()[0].GetPixels() == image.GetPixels());
        //naivebayes::Image image(0, 5, pixels);

    }
}