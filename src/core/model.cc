#include <core/model.h>
#include <iostream>
#include <iomanip>


namespace naivebayes {
    using namespace  std;

    Model::Model(size_t image_size): training_data_(TrainingData(image_size,0)) {
        image_size_ = image_size;

        pixel_probability_.resize(image_size_);
        for (size_t row = 0; row < image_size_; row++) {
            pixel_probability_[row].resize(image_size_);

            for(size_t col = 0; col < image_size_; col++) {
                pixel_probability_[row][col].resize(kNumOfClasses);

                for (size_t label = 0; label < kNumOfClasses; label++) {
                    pixel_probability_[row][col][label].resize(kNumOfShades);
                }
            }
        }
        CalculateProbabilities();
    }

    Model::Model(TrainingData &training_data) : training_data_(training_data) {
        image_size_ = training_data.GetImageSize();
        training_data_ = training_data;

        pixel_probability_.resize(image_size_);
        for (size_t row = 0; row < image_size_; row++) {
            pixel_probability_[row].resize(image_size_);
            
            for(size_t col = 0; col < image_size_; col++) {
                pixel_probability_[row][col].resize(kNumOfClasses);
                
                for (size_t label = 0; label < kNumOfClasses; label++) {
                    pixel_probability_[row][col][label].resize(kNumOfShades);
                }
            }
        }
        CalculateProbabilities();
    }


    void Model::CalculateProbabilities() {
        CalculateClassProbabilities();
        CalculatePixelProbabilities();
    }

    void Model::CalculateClassProbabilities() {
        for (size_t class_num = 0; class_num < kNumOfClasses; class_num++) {
            int label_count = 0;
            for (const Image& image : training_data_.GetImages()) {
                if (image.GetLabel() == class_num) {
                    label_count++;
                }
            }
            double numerator = kLaplaceSmoothing + label_count;
            double denominator = kNumOfClasses * kLaplaceSmoothing + training_data_.GetImageCount();
            class_probabilities_.push_back(numerator / denominator);
            //std::cout << std::setprecision(std::numeric_limits<double>::digits) << class_probabilities_[class_num]<< std::endl;
        }
    }

    void Model::CalculatePixelProbabilities() {
        vector<size_t> num_of_images_in_class_with_shade;
        for (size_t row = 0; row < training_data_.GetImageSize(); row++) {
            for (size_t col = 0; col < training_data_.GetImageSize(); col++) {
                for (size_t shade_num = 0; shade_num < kNumOfShades; shade_num++) {
                    num_of_images_in_class_with_shade.clear();

                    num_of_images_in_class_with_shade.resize(kNumOfClasses);

                    for (const Image& image : training_data_.GetImages()) {
                        if (image.getPixels()[row][col] == shade_num) {
                            num_of_images_in_class_with_shade[image.GetLabel()] += 1;
                        }
                    }

                    for (size_t label = 0; label < kNumOfClasses; label++) {
                        double numerator = kLaplaceSmoothing + num_of_images_in_class_with_shade[label];
                        double denominator = kNumOfShades * kLaplaceSmoothing + training_data_.GetNumberOfImagesInClass(label);

                        pixel_probability_[row][col][label][shade_num] = numerator / denominator;
                        //std::cout << std::setprecision(std::numeric_limits<double>::digits) << pixel_probability_[row][col][label][shade_num]<< std::endl;
                    }
                }
            }
        }
    }

    ostream &operator<<(ostream &output, Model &model) {
        for (size_t class_num = 0; class_num < model.kNumOfClasses; class_num++) {
            output << model.class_probabilities_[class_num] << endl;
        }
        for (size_t row = 0; row < model.training_data_.GetImageSize(); row++) {
            for (size_t col = 0; col < model.training_data_.GetImageSize(); col++) {
                for (size_t shade_num = 0; shade_num < model.kNumOfShades; shade_num++) {
                    for (size_t label = 0; label < model.kNumOfClasses; label++) {
                        output << model.pixel_probability_[row][col][label][shade_num] << endl;
                    }
                }
                
            }
        }
        
        return output;
    }

    istream &operator>>(istream &input, Model &model) {
        for (size_t class_num = 0; class_num < model.kNumOfClasses; class_num++) {
            input >> model.class_probabilities_[class_num];
        }
        for (size_t row = 0; row < model.training_data_.GetImageSize(); row++) {
            for (size_t col = 0; col < model.training_data_.GetImageSize(); col++) {
                for (size_t shade_num = 0; shade_num < model.kNumOfShades; shade_num++) {
                    for (size_t label = 0; label < model.kNumOfClasses; label++) {
                        input >> model.pixel_probability_[row][col][label][shade_num];
                    }
                }

            }
        }
        return input;
    }

}  // namespace naivebayes