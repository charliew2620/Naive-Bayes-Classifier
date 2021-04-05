#pragma once

#include <string>
#include <vector>
#include <core/image.h>

namespace naivebayes {
    using namespace std;

    class TrainingData {
    public:
        TrainingData(size_t image_size, size_t image_count);
        friend istream& operator >> (istream& input,TrainingData& data);

        vector<Image> GetImages() const;
        size_t GetImageCount() const;
        size_t GetImageSize() const;

    private:
        size_t image_size_;
        size_t image_count_;

        char kWhitePixel = ' ';

        vector<Image> images_;

    };

}  // namespace naivebayes
