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

        const vector<Image> &GetImages() const;
        size_t GetImageCount() const;
        size_t GetImageSize() const;
        size_t GetNumberOfImagesInClass(int label) const;

    private:
        size_t image_size_;
        size_t image_count_;

        char kWhitePixel = ' ';
        char kGrayPixel = '+';
        char kBlackPixel = '#';

        vector<Image> images_;
        
        vector<int> num_of_images_;
        
        vector<vector<int>> FillImageWithPixels(istream& input,TrainingData& data);
    };

}  // namespace naivebayes
