#include <catch2/catch.hpp>

#include <core/training_data.h>
#include <iostream>
#include <fstream>

TEST_CASE("Tests basic getters") {
    std::ifstream input("../../../data/5by5testfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;

    SECTION("Tests getImageCount") {
        REQUIRE(training_data.GetImageCount() == 3);
    }

    SECTION("Tests getImageSize") {
        REQUIRE(training_data.GetImageSize() == 5);
    }

    SECTION("Tests GetNumberOfImagesInClass") {
        REQUIRE(training_data.GetNumberOfImagesInClass(0) == 1);
        REQUIRE(training_data.GetNumberOfImagesInClass(1) == 1);
        REQUIRE(training_data.GetNumberOfImagesInClass(2) == 0);
        REQUIRE(training_data.GetNumberOfImagesInClass(3) == 0);
        REQUIRE(training_data.GetNumberOfImagesInClass(7) == 1);
    }
}

TEST_CASE("Tests operator >> and GetImages getter") {
    std::ifstream input("../../../data/5by5testfile.txt");
    naivebayes::TrainingData training_data(5, 3);
    input >> training_data;

    //This tests if every image in the file is parsed correctly by creating a new 2d vector
    //filled with values from 0-2 based on their shades and comparing it with the parsed images of
    // the training data

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
    }

    SECTION("Second Image") {
        std::vector<std::vector<int>> pixels
                {
                        {0, 0, 2, 0, 0},
                        {0, 0, 1, 0, 0},
                        {0, 0, 2, 0, 0},
                        {0, 0, 2, 0, 0},
                        {0, 0, 1, 0, 0}
                };
        naivebayes::Image image(1, 5, pixels);
        REQUIRE(training_data.GetImages()[1].GetLabel() == image.GetLabel());
        REQUIRE(training_data.GetImages()[1].GetImageSize() == image.GetImageSize());
        REQUIRE(training_data.GetImages()[1].GetPixels() == image.GetPixels());
    }

    SECTION("Third Image") {
        std::vector<std::vector<int>> pixels
                {
                        {1, 2, 2, 1, 2},
                        {0, 0, 0, 2, 0},
                        {0, 0, 2, 0, 0},
                        {0, 2, 0, 0, 0},
                        {1, 0, 0, 0, 0}
                };
        naivebayes::Image image(7, 5, pixels);
        REQUIRE(training_data.GetImages()[2].GetLabel() == image.GetLabel());
        REQUIRE(training_data.GetImages()[2].GetImageSize() == image.GetImageSize());
        REQUIRE(training_data.GetImages()[2].GetPixels() == image.GetPixels());
    }
}

TEST_CASE("Tests invalid character") {
    std::ifstream input("../../../data/invalidcharacter.txt");
    naivebayes::TrainingData training_data(3, 3);

    REQUIRE_THROWS_AS(input >> training_data, std::invalid_argument);
}