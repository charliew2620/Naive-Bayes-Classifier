#include <catch2/catch.hpp>

#include <core/training_data.h>
#include <iostream>
#include <fstream>

TEST_CASE("Tests GetImageSize") {
    std::ifstream input("../../../data/5by5testfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;
    
    REQUIRE(training_data.GetImages()[0].GetImageSize() == 5);
    REQUIRE(training_data.GetImages()[1].GetImageSize() == 5);
    REQUIRE(training_data.GetImages()[2].GetImageSize() == 5);
}

TEST_CASE("Tests GetPixels") {
    std::ifstream input("../../../data/5by5testfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;

    REQUIRE(training_data.GetImages()[0].GetPixels() == std::vector<std::vector<int>>
            {
                    {2, 2, 2, 1, 1},
                    {1, 0, 0, 0, 2},
                    {1, 0, 0, 0, 2},
                    {1, 0, 0, 0, 2},
                    {2, 2, 2, 2, 1}
            });
    REQUIRE(training_data.GetImages()[1].GetPixels() == std::vector<std::vector<int>>
                    {
                        {0, 0, 2, 0, 0},
                        {0, 0, 1, 0, 0},
                        {0, 0, 2, 0, 0},
                        {0, 0, 2, 0, 0},
                        {0, 0, 1, 0, 0}
                    });
    REQUIRE(training_data.GetImages()[2].GetPixels() == std::vector<std::vector<int>>
                    {
                        {1, 2, 2, 1, 2},
                        {0, 0, 0, 2, 0},
                        {0, 0, 2, 0, 0},
                        {0, 2, 0, 0, 0},
                        {1, 0, 0, 0, 0}
                    });
}

TEST_CASE("Tests GetLabels") {
    std::ifstream input("../../../data/5by5testfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;
    
    REQUIRE(training_data.GetImages()[0].GetLabel() == 0);
    REQUIRE(training_data.GetImages()[1].GetLabel() == 1);
    REQUIRE(training_data.GetImages()[2].GetLabel() == 7);
}