#pragma once

#include <string>
#include <vector>
#include <core/training_data.h>

namespace naivebayes {
    using namespace std;

    class Model {
    public:
        /**
         * Default constructor for Model passing in training data.
         * @param training_data passed in as parameter.
         */
        explicit Model(TrainingData &training_data);

        /**
         * Constructor created to be used to load in file.
         * @param image_size of the image passed in.
         */
        explicit Model(size_t image_size);

        /**
         * Operator to create file.
         * @param output ostream parameter.
         * @param model object passed in.
         * @return the output.
         */
        friend ostream &operator<<(ostream &output, Model &model);

        /**
         * Reads in a file with operator.
         * @param input passed in from istream.
         * @param model model object passed in.
         * @return the input.
         */
        friend istream &operator>>(istream &input, Model &model);

        // getters
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

        /**
         * Calls methods for class and pixel probabilities.
         */
        void CalculateProbabilities();

        /**
         * Calculates class probabilities.
         */
        void CalculateClassProbabilities();

        /**
         * Calcultes the pixel probabilities.
         */
        void CalculatePixelProbabilities();

        /**
         * Helper method for resizing the probability vector.
         */
        void ResizePixelProbabilityVector();

        /**
         * Helper method for outputting class probabilities.
         * @param output ostream passed in as parameter.
         * @param model Model object passed in as parameter.
         */
        void OutputClassProbabilities(ostream &output, Model &model);

        /**
         * Helper method for outputting pixel probabilities.
         * @param output ostream passed in as parameter.
         * @param model Model object passed in as parameter.
         */
        void OutputPixelProbabilities(ostream &output, Model &model);

        /**
         * Helper method for inputting class probabilities.
         * @param input istream passed in as parameter.
         * @param model Model object passed in as parameter.
         */
        void ReadInClassProbabilities(istream &input, Model &model);

        /**
         * Helper method for inputting pixel probabilities.
         * @param input istream passed in as parameter.
         * @param model Model object passed in as parameter.
         */
        void ReadInPixelProbabilities(istream &input, Model &model);
    };

}  // namespace naivebayes
