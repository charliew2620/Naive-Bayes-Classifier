#include <core/training_data.h>
#include <core/image.h>

namespace naivebayes {
    using namespace std;

    std::string TrainingData::GetBestClass() const {
        return "CS 126";
    }

    TrainingData::TrainingData(size_t image_size, size_t image_count) {
        image_size_ = image_size;
        image_count_ = image_count;
    }

    istream &operator>>(istream &input, TrainingData &data) {
        for (int i = 0; i < data.image_count_; i++) {
            int label;
            input >> label;
            input.get();

            vector<vector<bool>> pixels(data.image_size_);
            char pixel_character;
            // 2d vector of bool
            for (size_t row = 0; row < data.image_size_; row++) {
                pixels[row].resize(data.image_size_);

                for (size_t col = 0; col < data.image_size_; col++) {
                    // from input fill the 2d vector
                    input.get(pixel_character);
                    if (pixel_character == '#' || pixel_character == '+') {
                        pixels[row][col] = true;

                } else {
                        pixels[row][col] = false;
                    }

            // make an image object
            // push that object into data.images
                input.get();
            }
                Image image(label, data.image_size_, pixels);
                data.images_.push_back(image);
        }

    }

    /**
     * read image label
     * declare 2d vector pixel of bool of size image size x image size
     * loop read image into this vector
     * Image image(image_size, label, pixel)
     * images.push_back(images)
     */
}


}  // namespace naivebayes