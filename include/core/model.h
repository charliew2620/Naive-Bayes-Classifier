#pragma once

#include <string>
#include <vector>
#include <core/load_data.h>

namespace naivebayes {
    using std::vector;

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
        
        Model() = default;

        /**
         * Operator to create file.
         * @param output ostream parameter.
         * @param model object passed in.
         * @return the output.
         */
        friend std::ostream &operator<<(std::ostream &output, Model &model);

        /**
         * Reads in a file with operator.
         * @param input passed in from istream.
         * @param model model object passed in.
         * @return the input.
         */
        friend std::istream &operator>>(std::istream &input, Model &model);

        // getters
        const vector<double> &GetClassProbabilities() const;

        const vector<vector<vector<vector<double>>>> &GetPixelProbabilities() const;

        const TrainingData &GetTrainingData() const;
        
        double ComputeAccuracy(const vector<Image> &images);

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
        void TrainModel();

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
        void OutputClassProbabilities(std::ostream &output, Model &model);

        /**
         * Helper method for outputting pixel probabilities.
         * @param output ostream passed in as parameter.
         * @param model Model object passed in as parameter.
         */
        void OutputPixelProbabilities(std::ostream &output, Model &model);

        /**
         * Helper method for inputting class probabilities.
         * @param input istream passed in as parameter.
         * @param model Model object passed in as parameter.
         */
        void ReadInClassProbabilities(std::istream &input, Model &model);

        /**
         * Helper method for inputting pixel probabilities.
         * @param input istream passed in as parameter.
         * @param model Model object passed in as parameter.
         */
        void ReadInPixelProbabilities(std::istream &input, Model &model);


        int ClassifyImageWithLabel(const Image &image);

        void CalculateLikelihoodScores(const Image &image);

        void CheckAccuracy(const Image &image, int computed_label);

        double CalculateAccuracy() const;

        vector<double> image_likelihood_scores_;
        int correct_classification_ = 0;

        int FindLikeliestLabel(const Image &image);
    };

}  // namespace naivebayes
