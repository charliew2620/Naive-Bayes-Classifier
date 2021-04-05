#include <core/model.h>
#include <iostream>
#include <iomanip>


namespace naivebayes {
    using namespace  std;

    Model::Model(TrainingData &training_data) {
        image_size_ = training_data.GetImageSize();
        CalculateProbabilities(training_data);
    }


    void Model::CalculateProbabilities(TrainingData& training_data) {
        CalculateClassProbabilities(training_data);
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
            std::cout <<  std::setprecision(std::numeric_limits<double>::digits) << class_probabilities_[class_num]<< std::endl;
        }
    }

    void Model::CalculatePixelProbabilities(TrainingData &training_data) {
        for (size_t row = 0; row < training_data.GetImageSize(); row++) {
            for (size_t col = 0; col < training_data.GetImageSize(); col++){
                for (size_t shade_num = 0; shade_num < kNumOfShades; shade_num++) {
                    // make a new vector of the number of images in this class
                    
                    for (const Image& image : training_data.GetImages()) {
                        // increment the number of images for the image's class
                        // this value is used in the denominator
                        
                        if (shade_num == image.getPixels()[row][col]) {
                            // increment another counter for number of images for the image's class with that shade
                            // this value is used in the numerator
                        }
                    }
                }
            }
        }
//        vector<vector<double>> shaded_pixel_probabilities(image_size_, vector<double>(image_size_));
//
//        vector<vector<int>> pixels_shaded(image_size_, vector<int>(image_size_, 0));
//
//        // new vector for the number of images in each class.
//
//        for (auto image : images) {
//            for (size_t row = 0; row < image.getImageSize(); row++) {
//                for (size_t col = 0; col < image.getImageSize(); col++) {
//                    if (image.getPixels()[row][col]) {
//                        pixels_shaded[row][col]++;
//                    }
//                }
//            }
//            
//            // get the label of the image and increment the vector we added earlier
//        }
//
//        for (size_t row = 0; row < image_size_; row++) {
//            for (size_t col = 0; col < image_size_; col++) {
//                double numerator = kLaplaceSmoothing + pixels_shaded[row][col];
//                double denominator = (2 * kLaplaceSmoothing) + ;
//
//                shaded_pixel_probabilities[row][col] = numerator / denominator;
//            }
//            shaded_pixel_probabilities_.push_back(shaded_pixel_probabilities);
//        }

//        int pixels_shaded = 0;
//        for (auto & row : image.getPixels()) {
//            for (auto && col : row) {
//                if (col) {
//                    pixels_shaded++;
//                }
//
//            }
//        }
    }


//    istream &operator>>(istream &input, TrainingData &model) {
//        return input;
//    }
}  // namespace naivebayes