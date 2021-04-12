#include <core/model.h>
#include <iostream>
#include <iomanip>


namespace naivebayes {
    using std::vector;

    Model::Model(size_t image_size) : training_data_(TrainingData(image_size, 0)) {
        image_size_ = image_size;
        ResizePixelProbabilityVector();

        TrainModel();
    }

    Model::Model(TrainingData &training_data) : training_data_(training_data) {
        image_size_ = training_data.GetImageSize();
        ResizePixelProbabilityVector();

        TrainModel();
    }

    void Model::ResizePixelProbabilityVector() {
        pixel_probabilities_.resize(image_size_, vector<vector<vector<double>>>(image_size_,
                                                                                vector<vector<double>>(kNumOfClasses,
                                                                                                       vector<double>(
                                                                                                               kNumOfShades))));
    }

    void Model::TrainModel() {
        CalculateClassProbabilities();
        CalculatePixelProbabilities();
    }

    void Model::CalculateClassProbabilities() {
        for (size_t class_num = 0; class_num < kNumOfClasses; class_num++) {
            int label_count = 0;

            for (const Image &image : training_data_.GetImages()) {
                if (image.GetLabel() == class_num) {
                    label_count++;
                }
            }
            double numerator = kLaplaceSmoothing + label_count;
            double denominator = kNumOfClasses * kLaplaceSmoothing + training_data_.GetImageCount();
            class_probabilities_.push_back(numerator / denominator);
        }
    }

    void Model::CalculatePixelProbabilities() {
        vector<size_t> num_of_images_in_class_with_shade;

        for (size_t row = 0; row < image_size_; row++) {
            for (size_t col = 0; col < image_size_; col++) {
                for (size_t shade_num = 0; shade_num < kNumOfShades; shade_num++) {
                    num_of_images_in_class_with_shade.clear();
                    num_of_images_in_class_with_shade.resize(kNumOfClasses);

                    for (const Image &image : training_data_.GetImages()) {
                        // If the pixel of the image is equal to the shade, increments the vector's label count
                        // by one by calling the label of the current image
                        if (image.GetPixels()[row][col] == shade_num) {
                            num_of_images_in_class_with_shade[image.GetLabel()]++;
                        }
                    }
                    for (size_t label = 0; label < kNumOfClasses; label++) {
                        double numerator = kLaplaceSmoothing + num_of_images_in_class_with_shade[label];
                        double denominator =
                                kNumOfShades * kLaplaceSmoothing + training_data_.GetNumberOfImagesInClass(label);

                        pixel_probabilities_[row][col][label][shade_num] = numerator / denominator;
                    }
                }
            }
        }
    }

    std::ostream &operator<<(std::ostream &output, Model &model) {
        model.OutputClassProbabilities(output, model);
        model.OutputPixelProbabilities(output, model);
        return output;
    }

    void Model::OutputClassProbabilities(std::ostream &output, Model &model) {
        for (size_t class_num = 0; class_num < model.kNumOfClasses; class_num++) {
            output << std::setprecision(std::numeric_limits<double>::digits) << model.class_probabilities_[class_num]
                   << std::endl;
        }
    }

    void Model::OutputPixelProbabilities(std::ostream &output, Model &model) {
        output << model.training_data_.GetImageSize() << std::endl;
        for (size_t row = 0; row < model.image_size_; row++) {
            for (size_t col = 0; col < model.image_size_; col++) {
                for (size_t shade_num = 0; shade_num < model.kNumOfShades; shade_num++) {
                    for (size_t label = 0; label < model.kNumOfClasses; label++) {
                        output << std::setprecision(std::numeric_limits<double>::digits)
                               << model.pixel_probabilities_[row][col][label][shade_num] << std::endl;
                    }
                }
            }
        }
    }

    std::istream &operator>>(std::istream &input, Model &model) {
        model.ReadInClassProbabilities(input, model);
        model.ReadInPixelProbabilities(input, model);
        return input;
    }

    void Model::ReadInClassProbabilities(std::istream &input, Model &model) {
        for (size_t class_num = 0; class_num < model.kNumOfClasses; class_num++) {
            input >> model.class_probabilities_[class_num];
        }
    }

    void Model::ReadInPixelProbabilities(std::istream &input, Model &model) {
        for (size_t row = 0; row < model.image_size_; row++) {
            for (size_t col = 0; col < model.image_size_; col++) {
                for (size_t shade_num = 0; shade_num < model.kNumOfShades; shade_num++) {
                    for (size_t label = 0; label < model.kNumOfClasses; label++) {
                        input >> model.pixel_probabilities_[row][col][label][shade_num];
                    }
                }
            }
        }
    }

    const vector<double> &Model::GetClassProbabilities() const {
        return class_probabilities_;
    }

    const vector<vector<vector<vector<double>>>> &Model::GetPixelProbabilities() const {
        return pixel_probabilities_;
    }

    const TrainingData &Model::GetTrainingData() const {
        return training_data_;
    }

    int Model::ClassifyImageWithLabel(const Image &image) {
        double max = -DBL_MAX;
        int best_label;
        for (size_t label = 0; label < image_likelihood_scores_.size(); label++) {
            if (image_likelihood_scores_[label] > max) {
                max = image_likelihood_scores_[label];
                best_label = label;
            }
        }
        return best_label;
    }

    void Model::CalculateLikelihoodScores(const Image &image) {
        image_likelihood_scores_.clear();
        for (double class_probability : class_probabilities_) {
            image_likelihood_scores_.push_back(log(class_probability));
        }
        
        for (size_t label = 0; label < image_likelihood_scores_.size(); label++) {
            for (size_t row = 0; row < image.GetImageSize(); row++) {
                for (size_t col = 0; col < image.GetImageSize(); col++) {
                    image_likelihood_scores_[label] += log(
                           pixel_probabilities_[row][col][label][image.GetPixels()[row][col]]);
                }
            }
        }
    }

    void Model::CheckAccuracy(const Image &image, int computed_label) {
        if (image.GetLabel() == computed_label) {
            correct_classification_++;
        } else {
            //std::cout << computed_label << " " << image.GetLabel() << std::endl;
        }
    }

    double Model::CalculateAccuracy() const {
        return (double) correct_classification_ / 1000;
    }

    double Model::ComputeAccuracy(const vector<Image> &images) {
        for (const Image &image : images) {
            FindLikeliestLabel(image);
        }
        double a = CalculateAccuracy();
        return a;
    }
    
    int Model::FindLikeliestLabel(const Image& image){
        CalculateLikelihoodScores(image);
        CheckAccuracy(image, ClassifyImageWithLabel(image));
        return ClassifyImageWithLabel(image);
    }
}  // namespace naivebayes