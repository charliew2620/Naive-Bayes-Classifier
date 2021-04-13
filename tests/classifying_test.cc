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

TEST_CASE("Tests accuracy of smaller test file") {
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

TEST_CASE("Tests likelihood scores and label function of 28x28 file with 3 images") {
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
    model.ComputeAccuracy(testing_data.GetImages());

    SECTION("Tests last image of label 6 for likelihood scores") {
        REQUIRE(model.GetLikelihoodScores().size() == 10);
        REQUIRE(model.GetLikelihoodScores()[0] == Approx(-455.366).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[1] == Approx(-568.170).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[2] == Approx(-352.840).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[3] == Approx(-441.909).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[4] == Approx(-387.442).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[5] == Approx(-388.431).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[6] == Approx(-255.747).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[7] == Approx(-478.668).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[8] == Approx(-383.424).epsilon(0.01));
        REQUIRE(model.GetLikelihoodScores()[9] == Approx(-406.066).epsilon(0.01));
    }

    SECTION("Tests FindLikeliestLabel function") {
        for (int image = 0; image < 3; image++) {
            REQUIRE(model.FindLikeliestLabel(testing_data.GetImages()[image].GetPixels()) ==
                    testing_data.GetImages()[image].GetLabel());
        }
    }
    os.close();
}

TEST_CASE("Tests likelihood scores for 3x3 file with 3 images") {
    std::ifstream stream("../../../data/trainingimagesandlabels.txt");
    naivebayes::LoadData training_data(28, 5000);
    stream >> training_data;

    naivebayes::Model model(training_data);

    std::ofstream os;
    os.open("../../../data/wheredidmyweekendgo.txt");
    os << model;


    std::ifstream stream1("../../../data/3by3testfile.txt");
    naivebayes::LoadData testing_data(3, 3);
    stream1 >> testing_data;
    model.ComputeAccuracy(testing_data.GetImages());

    REQUIRE(model.GetLikelihoodScores()[0] == Approx(-33.252).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[1] == Approx(-33.891).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[2] == Approx(-33.325).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[3] == Approx(-33.366).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[4] == Approx(-33.689).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[5] == Approx(-32.862).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[6] == Approx(-33.429).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[7] == Approx( -33.799).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[8] == Approx( -33.109).epsilon(0.01));
    REQUIRE(model.GetLikelihoodScores()[9] == Approx(-33.382).epsilon(0.01));
}

