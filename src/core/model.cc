#include <core/model.h>


namespace naivebayes {
    using namespace  std;

    Model::Model(TrainingData &training_data) {
        CalculateProbabilities(training_data);
    }


    void Model::CalculateProbabilities(TrainingData& training_data) {
        for (Image images : training_data.getImages()) {

        }
    }

    void Model::CalculateClassProbabilities(TrainingData &training_data) {


    }

    void Model::CalculatePixelProbabilities(vector<Image> images) {

    }


//    istream &operator>>(istream &input, TrainingData &model) {
//        return input;
//    }
}  // namespace naivebayes