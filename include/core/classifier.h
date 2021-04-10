#pragma once

#include <string>
#include <vector>
#include "model.h"

namespace naivebayes {
    using std::vector;

    class Classifier {
    public:
        
        explicit Classifier(Model model);
        
    private:
        Model model_;
        vector<double> image_likelihood_scores_;
        int correct_classification_ = 0;
        int total_images_ = 0;

        int ClassifyImageWithLabel(Image &image);
        
        void CalculateLikelihoodScores(Image &image);
        
        void CheckAccuracy(Image &image, int computed_label);
        
        double CalculateAccuracy() const;
        
    };

}  // namespace naivebayes
