#include <core/model.h>
#include <iostream>
#include <iomanip>


namespace naivebayes {
    using namespace  std;

    Model::Model(size_t image_size): training_data_(TrainingData(image_size, 0)) {
        image_size_ = image_size;
        ResizePixelProbabilityVector();
        
        CalculateProbabilities();
    }

    Model::Model(TrainingData &training_data) : training_data_(training_data) {
        image_size_ = training_data.GetImageSize();
        training_data_ = training_data;
        ResizePixelProbabilityVector();
        
        CalculateProbabilities();
    }

    void Model::ResizePixelProbabilityVector() {
        pixel_probabilities_.resize(image_size_);
        for (size_t row = 0; row < image_size_; row++) {
            pixel_probabilities_[row].resize(image_size_);

            for(size_t col = 0; col < image_size_; col++) {
                pixel_probabilities_[row][col].resize(kNumOfClasses);

                for (size_t label = 0; label < kNumOfClasses; label++) {
                    pixel_probabilities_[row][col][label].resize(kNumOfShades);
                }
            }
        }
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
        
        for (size_t row = 0; row < image_size_; row++) {
            for (size_t col = 0; col < image_size_; col++) {
                for (size_t shade_num = 0; shade_num < kNumOfShades; shade_num++) {
                    num_of_images_in_class_with_shade.clear();
                    num_of_images_in_class_with_shade.resize(kNumOfClasses);

                    for (const Image& image : training_data_.GetImages()) {
                        if (image.GetPixels()[row][col] == shade_num) {
                            num_of_images_in_class_with_shade[image.GetLabel()]++;
                        }
                    }
                    for (size_t label = 0; label < kNumOfClasses; label++) {
                        double numerator = kLaplaceSmoothing + num_of_images_in_class_with_shade[label];
                        double denominator = kNumOfShades * kLaplaceSmoothing + training_data_.GetNumberOfImagesInClass(label);

                        pixel_probabilities_[row][col][label][shade_num] = numerator / denominator;
                        //std::cout << std::setprecision(std::numeric_limits<double>::digits) << pixel_probability_[row][col][label][shade_num]<< std::endl;
                    }
                }
            }
        }
    }

    ostream &operator<<(ostream &output, Model &model) {
        model.OutputClassProbabilities(output, model);
        model.OutputPixelProbabilities(output, model);
        return output;
    }

    void Model::OutputClassProbabilities(ostream &output, Model &model) {
        for (size_t class_num = 0; class_num < model.kNumOfClasses; class_num++) {
            output << std::setprecision(std::numeric_limits<double>::digits) << model.class_probabilities_[class_num] << endl;
        }
    }

    void Model::OutputPixelProbabilities(ostream &output, Model &model) {
        for (size_t row = 0; row < model.image_size_; row++) {
            for (size_t col = 0; col < model.image_size_; col++) {
                for (size_t shade_num = 0; shade_num < model.kNumOfShades; shade_num++) {
                    for (size_t label = 0; label < model.kNumOfClasses; label++) {
                        output << std::setprecision(std::numeric_limits<double>::digits) << model.pixel_probabilities_[row][col][label][shade_num] << endl;
                    }
                }
            }
        }
    }

    istream &operator>>(istream &input, Model &model) {
        model.ReadInClassProbabilities(input, model);
        model.ReadInPixelProbabilities(input, model);
        return input;
    }

    void Model::ReadInClassProbabilities(istream &input, Model &model) {
        for (size_t class_num = 0; class_num < model.kNumOfClasses; class_num++) {
            input >> model.class_probabilities_[class_num];
        }
    }

    void Model::ReadInPixelProbabilities(istream &input, Model &model) {
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
}  // namespace naivebayes