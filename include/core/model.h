#include <string>
#include <vector>
#include <core/training_data.h>

namespace naivebayes {
    using namespace std;

    class Model {
    public:

        explicit Model(TrainingData &training_data);
        
        explicit Model(size_t image_size);

        friend ostream& operator << (ostream& output, Model& model);
        
        friend istream& operator >> (istream &input, Model& model);

        const vector<double> &GetClassProbabilities() const;
        const vector<vector<vector<vector<double>>>> &GetPixelProbabilities() const;
        const TrainingData &GetTrainingData() const; 
        

    private:
        const double kLaplaceSmoothing = 1;
        const size_t kNumOfClasses = 10;
        const size_t kNumOfShades = 3;

        TrainingData training_data_;

        size_t image_size_;

        vector<double> class_probabilities_;
        vector<vector<vector<vector<double>>>> pixel_probabilities_;

        void CalculateProbabilities();
        void CalculateClassProbabilities();
        void CalculatePixelProbabilities();
        
        void ResizePixelProbabilityVector();
        
        void OutputClassProbabilities(ostream& output, Model &model);
        void OutputPixelProbabilities(ostream& output, Model &model);

        void ReadInClassProbabilities(istream& input, Model &model);
        void ReadInPixelProbabilities(istream& input, Model &model);
    };

}  // namespace naivebayes
