#include <core/training_data.h>

namespace naivebayes {
    using namespace  std;

std::string TrainingData::GetBestClass() const {
  return "CS 126";
}

    map<int, vector<vector<double>>>
    TrainingData::calculate_image_probability(vector<vector<float>> data, float training_label) {
        return map<int, vector<vector<double>>>();
    }

    map<int, double>
    TrainingData::calculate_class_probability(map<int, vector<vector<float>>> data, float training_label) {
        return map<int, double>();
    }

}  // namespace naivebayes