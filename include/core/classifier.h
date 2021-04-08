#pragma once

#include <string>
#include <vector>
#include "model.h"

namespace naivebayes {
    using namespace std;

    class Classifier {
    public:
        
        explicit Classifier(Model model);
        
    private:
        Model model_;
        vector<double> image_likelihood_scores_;

        int ClassifyImageWithLabel(Image &image);
        
        void CalculateLikelihoodScores(Image &image);
        
    };

}  // namespace naivebayes
