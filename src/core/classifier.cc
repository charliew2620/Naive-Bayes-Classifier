#include <utility>
#include <core/classifier.h>

namespace naivebayes {
    using namespace std;

    Classifier::Classifier(Model model) : model_(std::move(model)) {

    }

    int Classifier::ClassifyImageWithLabel(Image &image) {
        return 0;
    }

    void Classifier::CalculateLikelihoodScores(Image &image) {
        for (double class_probability : model_.GetClassProbabilities()) {
            image_likelihood_scores_.push_back(log(class_probability));
        }
        
        for (size_t row = 0; row < image.GetImageSize(); row++) {
            for (size_t col = 0; col < image.GetImageSize(); col++) {
                for (size_t shade = 0; shade < model_.GetPixelProbabilities().size(); shade++) {
                    if (image.GetPixels()[row][col] == 0) {
                        //unshaded
                        continue;
                    } else if (image.GetPixels()[row][col] == 1) {
                        //partially shaded
                        continue;
                    } else {
                        // shaded
                        continue;
                    }
                }
            }
        }
    }

}  // namespace naivebayes