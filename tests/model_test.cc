#include <catch2/catch.hpp>

#include <core/load_data.h>
#include <core/model.h>
#include <iostream>
#include <fstream>

TEST_CASE("Tests class probabilities") {
    std::ifstream input("../../../data/5by5testfile.txt");
    naivebayes::LoadData training_data(5, 3);
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

    SECTION("Checks probability for class 7") {
        REQUIRE(model.GetClassProbabilities()[7] == Approx(0.1538).epsilon(0.01));
    }

    SECTION("Checks probability for classes not in file") {
        for (int class_num = 0; class_num < 10; class_num++) {
            if (class_num != 0 && class_num != 1 && class_num != 7) {
                REQUIRE(model.GetClassProbabilities()[class_num] == Approx(0.0769).epsilon(0.01));
            }
        }
    }
}

TEST_CASE("Tests pixel probabilities size") {
    std::ifstream input("../../../data/3by3testfile.txt");
    naivebayes::LoadData training_data(3, 3);
    input >> training_data;
    naivebayes::Model model(training_data);

    SECTION("Checks row size") {
        REQUIRE(model.GetPixelProbabilities().size() == 3);
    }

    SECTION("Checks column size") {
        REQUIRE(model.GetPixelProbabilities()[0].size() == 3);
    }

    SECTION("Checks label size") {
        REQUIRE(model.GetPixelProbabilities()[0][0].size() == 10);
    }

    SECTION("Checks shade size") {
        REQUIRE(model.GetPixelProbabilities()[0][0][0].size() == 3);
    }
}

TEST_CASE("Tests pixel probabilities") {
    std::ifstream input("../../../data/3by3testfile.txt");
    naivebayes::LoadData training_data(3, 3);
    input >> training_data;
    naivebayes::Model model(training_data);

    SECTION("Tests probabilities for label 0 no shade") {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (row != 1 && col != 1) {
                    REQUIRE(model.GetPixelProbabilities()[row][col][0][0] == Approx(0.25).epsilon(0.01));
                }
            }
        }
        REQUIRE(model.GetPixelProbabilities()[1][1][0][0] == Approx(0.5).epsilon(0.01));
    }

    SECTION("Tests probabilities for label 0 partial shade") {
        REQUIRE(model.GetPixelProbabilities()[0][0][0][1] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[0][1][0][1] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[0][2][0][1] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][0][0][1] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][1][0][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][2][0][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][0][0][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][1][0][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][2][0][1] == Approx(0.5).epsilon(0.01));
    }

    SECTION("Tests probabilities for label 0 total shade") {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (row == 1 && col == 2 || row == 2 && col == 0 || row == 2 && col == 1) {
                    REQUIRE(model.GetPixelProbabilities()[row][col][0][2] == Approx(0.5).epsilon(0.01));
                } else {
                    REQUIRE(model.GetPixelProbabilities()[row][col][0][2] == Approx(0.25).epsilon(0.01));
                }
            }
        }
    }

    SECTION("Tests probabilities for label 1 no shade") {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (col == 0 || col == 2) {
                    REQUIRE(model.GetPixelProbabilities()[row][col][1][0] == Approx(0.5).epsilon(0.01));
                } else if (col == 1) {
                    REQUIRE(model.GetPixelProbabilities()[row][col][1][0] == Approx(0.25).epsilon(0.01));
                }
            }
        }
    }

    SECTION("Tests probabilities for label 1 partial shade") {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (row == 1 && col == 1 || row == 2 && col == 1) {
                    REQUIRE(model.GetPixelProbabilities()[row][col][1][1] == Approx(0.5).epsilon(0.01));
                } else {
                    REQUIRE(model.GetPixelProbabilities()[row][col][1][1] == Approx(0.25).epsilon(0.01));
                }
            }
        }
    }

    SECTION("Tests probabilities for label 1 total shade") {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (row == 0 && col == 1) {
                    REQUIRE(model.GetPixelProbabilities()[row][col][1][2] == Approx(0.5).epsilon(0.01));
                } else {
                    REQUIRE(model.GetPixelProbabilities()[row][col][1][2] == Approx(0.25).epsilon(0.01));
                }
            }
        }
    }

    SECTION("Tests probabilities for label 7 no shade") {
        REQUIRE(model.GetPixelProbabilities()[0][0][7][0] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[0][1][7][0] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[0][2][7][0] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][0][7][0] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][1][7][0] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][2][7][0] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][0][7][0] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][1][7][0] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][2][7][0] == Approx(0.5).epsilon(0.01));
    }

    SECTION("Tests probabilities for label 7 partial shade") {
        REQUIRE(model.GetPixelProbabilities()[0][0][7][1] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[0][1][7][1] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[0][2][7][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][0][7][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][1][7][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[1][2][7][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][0][7][1] == Approx(0.5).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][1][7][1] == Approx(0.25).epsilon(0.01));
        REQUIRE(model.GetPixelProbabilities()[2][2][7][1] == Approx(0.25).epsilon(0.01));
    }

    SECTION("Tests probabilities for label 7 total shade") {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (row == 0 && col == 2 || row == 1 && col ==1) {
                    REQUIRE(model.GetPixelProbabilities()[row][col][7][2] == Approx(0.5).epsilon(0.01));
                } else {
                    REQUIRE(model.GetPixelProbabilities()[row][col][7][2] == Approx(0.25).epsilon(0.01));
                }
            }
        }
    }

    SECTION("Tests probabilities for nonexistent label no shade") {
        for (int label = 0; label < 10; label++) {
            if (label != 0 && label != 1 && label != 7) {
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 3; col++) {
                        REQUIRE(model.GetPixelProbabilities()[row][col][label][0] == Approx(0.3333).epsilon(0.01));
                    }
                }
            }
        }
    }

    SECTION("Tests probabilities for nonexistent label partial shade") {
        for (int label = 0; label < 10; label++) {
            if (label != 0 && label != 1 && label != 7) {
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 3; col++) {
                        REQUIRE(model.GetPixelProbabilities()[row][col][label][1] == Approx(0.3333).epsilon(0.01));
                    }
                }
            }
        }
    }

    SECTION("Tests probabilities for nonexistent label total shade") {
        for (int label = 0; label < 10; label++) {
            if (label != 0 && label != 1 && label != 7) {
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 3; col++) {
                        REQUIRE(model.GetPixelProbabilities()[row][col][label][2] == Approx(0.3333).epsilon(0.01));
                    }
                }
            }
        }
    }
}

TEST_CASE("Test saving and loading model") {
    // Creates a model
    std::ifstream stream("../../../data/5by5testfile.txt");
    naivebayes::LoadData training_data(5, 3);
    stream >> training_data;

    naivebayes::Model model(training_data);

    // Saves the model
    std::ofstream os;
    os.open("../../../data/testsave.txt");
    os << model;
    os.close();

    // Loads the model again
    std::ifstream ifstream("../../../data/testsave.txt");
    ifstream >> model;


    // Checks if all images are loaded correctly
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
        REQUIRE(model.GetLoadData().GetImages()[0].GetLabel() == image.GetLabel());
        REQUIRE(model.GetLoadData().GetImages()[0].GetImageSize() == image.GetImageSize());
        REQUIRE(model.GetLoadData().GetImages()[0].GetPixels() == image.GetPixels());
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
        REQUIRE(model.GetLoadData().GetImages()[1].GetLabel() == image.GetLabel());
        REQUIRE(model.GetLoadData().GetImages()[1].GetImageSize() == image.GetImageSize());
        REQUIRE(model.GetLoadData().GetImages()[1].GetPixels() == image.GetPixels());
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
        REQUIRE(model.GetLoadData().GetImages()[2].GetLabel() == image.GetLabel());
        REQUIRE(model.GetLoadData().GetImages()[2].GetImageSize() == image.GetImageSize());
        REQUIRE(model.GetLoadData().GetImages()[2].GetPixels() == image.GetPixels());
    }
}
