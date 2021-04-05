#include <core/model.h>
#include <iostream>
#include <iomanip>


namespace naivebayes {
    using namespace  std;

    Model::Model(TrainingData &training_data) {
        image_size_ = training_data.GetImageSize();
        
        pixel_probability_.resize(image_size_);
        for (size_t i = 0; i < image_size_; i++) {
            pixel_probability_[i].resize(image_size_);
            
            for(size_t j = 0; j < image_size_; j++) {
                pixel_probability_[i][j].resize(kNumOfClasses);
                
                for (size_t k = 0; k < kNumOfClasses; k++) {
                    pixel_probability_[i][j][k].resize(kNumOfShades);
                }
            }
        }
        
        CalculateProbabilities(training_data);
    }


    void Model::CalculateProbabilities(TrainingData& training_data) {
        CalculateClassProbabilities(training_data);
        CalculatePixelProbabilities(training_data);
    }

    void Model::CalculateClassProbabilities(TrainingData &training_data) {
        
        for (size_t class_num = 0; class_num < kNumOfClasses; class_num++) {
            int label_count = 0;
            for (const Image& image : training_data.GetImages()) {
                if (image.GetLabel() == class_num) {
                    label_count++;
                }
            }
            double numerator = kLaplaceSmoothing + label_count;
            double denominator = kNumOfClasses * kLaplaceSmoothing + training_data.GetImageCount();
            class_probabilities_.push_back(numerator / denominator);
            //std::cout <<  std::setprecision(std::numeric_limits<double>::digits) << class_probabilities_[class_num]<< std::endl;
        }
    }

    void Model::CalculatePixelProbabilities(TrainingData &training_data) {
        
        for (size_t row = 0; row < training_data.GetImageSize(); row++) {
            for (size_t col = 0; col < training_data.GetImageSize(); col++){
                for (size_t shade_num = 0; shade_num < kNumOfShades; shade_num++) {
                    vector<size_t> num_of_images_in_class_with_shade;
                    
                    num_of_images_in_class_with_shade.resize(kNumOfClasses);
                    
                    for (const Image& image : training_data.GetImages()) {
                        if (image.getPixels()[row][col] == shade_num) {
                            num_of_images_in_class_with_shade[image.GetLabel()] += 1;
                        }
                    }
                    
                    for (size_t label = 0; label < kNumOfClasses; label++) {
                        double numerator = kLaplaceSmoothing + num_of_images_in_class_with_shade[label];
                        double denominator = kNumOfShades * kLaplaceSmoothing + training_data.GetNumberOfImagesInClass(label);
                        
                        pixel_probability_[row][col][label][shade_num] = numerator / denominator;
                        std::cout << std::setprecision(std::numeric_limits<double>::digits) << pixel_probability_[row][col][label][shade_num]<< std::endl;
                    }
                }
            }
        }
    }


//    istream &operator>>(istream &input, TrainingData &model) {
//        return input;
//    }
}  // namespace naivebayes