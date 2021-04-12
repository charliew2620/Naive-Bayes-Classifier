#include <core/load_data.h>

namespace naivebayes {
    using std::vector;
    using std::string;

    LoadData::LoadData(size_t image_size, size_t image_count) {
        image_size_ = image_size;
        image_count_ = image_count;
        num_of_images_in_class_.resize(kMagicTen);
    }

    std::istream &operator>>(std::istream &input, LoadData &data) {
        for (size_t i = 0; i < data.image_count_; i++) {
            int label;
            input >> label;
            // gets the label of the image from the file
            input.get();

            vector<vector<int>> pixels = data.FillImageWithPixels(input, data);

            // index at this vector incremented for getter
            data.num_of_images_in_class_[label]++;

            data.num_of_images_per_label_.emplace_back(label, data.image_size_, pixels);
        }
        return input;
    }

    vector<vector<int>> LoadData::FillImageWithPixels(std::istream &input, LoadData &data) {
        vector<vector<int>> pixels(data.image_size_);
        char pixel_character;

        for (size_t row = 0; row < data.image_size_; row++) {
            pixels[row].resize(data.image_size_);

            string line;
            getline(input, line);

            // Fills 2d vector with number values for shades
            for (size_t col = 0; col < data.image_size_; col++) {
                pixel_character = line[col];
                if (pixel_character == data.kWhitePixel) {
                    pixels[row][col] = kWhitePixelValue;

                } else if (pixel_character == data.kGrayPixel) {
                    pixels[row][col] = kGrayPixelValue;

                } else if (pixel_character == data.kBlackPixel) {
                    pixels[row][col] = kBlackPixelValue;

                } else {
                    throw std::invalid_argument("Invalid character");
                }
            }
        }
        return pixels;
    }

    const size_t &LoadData::GetImageCount() const {
        return image_count_;
    }

    const size_t &LoadData::GetImageSize() const {
        return image_size_;
    }

    size_t LoadData::GetNumberOfImagesInClass(int label) const {
        return num_of_images_in_class_[label];
    }

    const vector<Image> &LoadData::GetImages() const {
        return num_of_images_per_label_;
    }
}  // namespace naivebayes