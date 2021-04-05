#include <string>
#include <vector>
#include <core/training_data.h>

namespace naivebayes {
    using namespace std;

    class Model {
    public:

        explicit Model(TrainingData &training_data);

        friend ostream& operator << (ostream& output, Model& model);

    private:
        const double kLaplaceSmoothing = 1;
        const size_t kNumOfClasses = 10;
        const size_t kNumOfShades = 3;

        TrainingData& training_data_;

        size_t image_size_;

        vector<double> class_probabilities_;
        vector<vector<vector<vector<double>>>> pixel_probability_;

        void CalculateProbabilities();
        void CalculateClassProbabilities();
        void CalculatePixelProbabilities();

    };

}  // namespace naivebayes
