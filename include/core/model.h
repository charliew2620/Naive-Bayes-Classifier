#include <string>
#include <vector>
#include <core/training_data.h>

namespace naivebayes {
    using namespace std;

    class Model {
    public:

        explicit Model(TrainingData& training_data);

    private:
        const int kLaplaceSmoothing = 1;
        const int kNumOfClasses = 10;


        vector<double> class_probabilities_;
        vector<vector<vector<double>>> shaded_pixel_probabilities_;

        void CalculateProbabilities(TrainingData& training_data);
        void CalculateClassProbabilities(TrainingData& training_data);
        void CalculatePixelProbabilities(vector<Image> images);

    };

}  // namespace naivebayes
