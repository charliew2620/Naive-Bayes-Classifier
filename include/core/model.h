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
        
//        Model() = default;

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

        /**
         * Calculates and finds the label with the highest probability for a passed image.
         * @param image to be assigned a label.
         * @return predicted label of image.
         */
        int FindLikeliestLabel(const vector<vector<int>>& image);

        /**
         * Calculates and returns the accuracy of the classifier.
         * @param images passed to determine accuracy.
         * @return a decimal for accuracy.
         */
        double ComputeAccuracy(const vector<Image> &images);

        // getters
        const vector<double> &GetClassProbabilities() const;

        const vector<vector<vector<vector<double>>>> &GetPixelProbabilities() const;

    private:
        const double kLaplaceSmoothing = 1;
        const size_t kNumOfClasses = 10;
        const size_t kNumOfShades = 3;

        TrainingData training_data_;

        size_t image_size_;

        vector<double> class_probabilities_;
        vector<vector<vector<vector<double>>>> pixel_probabilities_;

        vector<double> image_likelihood_scores_;
        int correct_classification_ = 0;

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

        /**
         * Goes through all the probabilities and assigns the image with the label of highest probability.
         * @param image to be classified.
         * @return the most likely label.
         */
        int ClassifyImageWithLabel(const vector<vector<int>>& image);

        /**
         * Calculates the likelihood scores of every label using equation given by document.
         * @param image passed for calculating likelihood scores.
         */
        void CalculateLikelihoodScores(const vector<vector<int>>& image);

        /**
         * Checks of the predicted label matches the actual label of the image.
         * @param image to be checked to see if label is correct.
         * @param computed_label the predicted label to be compare with image's actual label.
         */
        void CheckAccuracy(const Image& image, int computed_label);
    };

}  // namespace naivebayes
