#include <utility>
#include <core/classifier.h>
#include <iostream>

namespace naivebayes {
    using std::vector;

    Classifier::Classifier(Model model) : model_(std::move(model)) {
        for (Image image : model_.GetTrainingData().GetImages()) {
            CalculateLikelihoodScores(image);
            ClassifyImageWithLabel(image);
            CheckAccuracy(image, ClassifyImageWithLabel(image));
        }
        CalculateAccuracy();
    }

    int Classifier::ClassifyImageWithLabel(Image &image) {
        double max = -INT_MAX;
        int best_label;
        for (size_t label = 0; label < image_likelihood_scores_.size(); label++) {
            if (image_likelihood_scores_[label] > max) {
                max = image_likelihood_scores_[label];
                best_label = label;
            }
        }
        return best_label;
    }

    void Classifier::CalculateLikelihoodScores(Image &image) {
        image_likelihood_scores_.clear();
        for (double class_probability : model_.GetClassProbabilities()) {
            image_likelihood_scores_.push_back(log(class_probability));
        }

        for (size_t label = 0; label < image_likelihood_scores_.size(); label++) {
            for (size_t row = 0; row < image.GetImageSize(); row++) {
                for (size_t col = 0; col < image.GetImageSize(); col++) {
                    image_likelihood_scores_[label] += log(
                            model_.GetPixelProbabilities()[row][col][label][image.GetPixels()[row][col]]);
                }
            }
        }
    }

    void Classifier::CheckAccuracy(Image &image, int computed_label) {
        if (image.GetLabel() == computed_label) {
            correct_classification_++;
        }
        total_images_++;
    }

    double Classifier::CalculateAccuracy() const {
        std::cout << correct_classification_ << std::endl;
        std::cout << total_images_ << std::endl;
        std:: cout << (double) correct_classification_ / total_images_;
        return (double) correct_classification_ / total_images_;
    }
}  // namespace naivebayes