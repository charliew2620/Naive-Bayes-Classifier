#include <string>
#include <vector>
#include <core/training_data.h>

namespace naivebayes {
    using namespace std;

    class Model {
    public:

        explicit Model(TrainingData& training_data);

    private:
        const double kLaplaceSmoothing = 1;
        const size_t kNumOfClasses = 10;
        const size_t kNumOfShades = 2;
        
        const int kNumofPixels = 784;

        int image_size_;

        vector<double> class_probabilities_;
        vector<vector<vector<vector<double>>>> pixel_probability_;
        
        vector<vector<vector<double>>> shaded_pixel_probabilities_;

        void CalculateProbabilities(TrainingData& training_data);
        void CalculateClassProbabilities(TrainingData& training_data);
        void CalculatePixelProbabilities(TrainingData &training_data);

    };

}  // namespace naivebayes
