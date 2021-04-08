#include <utility>
#include <core/classifier.h>

namespace naivebayes {
    using namespace std;

    Classifier::Classifier(Model model) : model_(std::move(model)) {

    }

    int Classifier::ClassifyImageWithLabel(Image &image) {
        double max = -1;
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
        for (double class_probability : model_.GetClassProbabilities()) {
            image_likelihood_scores_.push_back(log(class_probability));
        }
        
        for (size_t label = 0; label < image_likelihood_scores_.size(); label++) {
            for (size_t row = 0; row < image.GetImageSize(); row++) {
                for (size_t col = 0; col < image.GetImageSize(); col++) {
                    for (size_t shade = 0; shade < 3; shade++) {
                        image_likelihood_scores_[label] += log(model_.GetPixelProbabilities()[row][col][label][shade]);
                    }
                }
            }
        }
    }

}  // namespace naivebayes