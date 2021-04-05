#include <core/training_data.h>

namespace naivebayes {
    using namespace std;

    TrainingData::TrainingData(size_t image_size, size_t image_count) {
        image_size_ = image_size;
        image_count_ = image_count;
        
        num_of_images_.resize(10);
    }

    istream &operator>>(istream &input, TrainingData &data) {
        for (size_t i = 0; i < data.image_count_; i++) {
            int label;
            input >> label;
            input.get();

            vector<vector<int>> pixels = data.FillImageWithPixels(input, data);
            
            data.num_of_images_[label]++;
            data.images_.emplace_back(label, data.image_size_, pixels);
        }
        return input;
    }

    vector<vector<int>> TrainingData::FillImageWithPixels(istream &input, TrainingData &data) {
        vector<vector<int>> pixels(data.image_size_);
        char pixel_character;

        for (size_t row = 0; row < data.image_size_; row++) {
            pixels[row].resize(data.image_size_);

            string line;
            getline(input, line);

            // Fills 2d vector with boolean values for shades
            for (size_t col = 0; col < data.image_size_; col++) {
                pixel_character = line[col];
                if (pixel_character == data.kWhitePixel) {
                    pixels[row][col] = 0;

                } else if (pixel_character == data.kGrayPixel) {
                    pixels[row][col] = 1;
                } else if (pixel_character == data.kBlackPixel) {
                    pixels[row][col] = 2;
                }
            }
        }
        return pixels;
    }
    
    size_t TrainingData::GetImageCount() const {
        return image_count_;
    }

    size_t TrainingData::GetImageSize() const {
        return image_size_;
    }

    size_t TrainingData::GetNumberOfImagesInClass(int label) const {
        return num_of_images_[label];
    }

    const vector<Image> &TrainingData::GetImages() const {
        return images_;
    }
}  // namespace naivebayes