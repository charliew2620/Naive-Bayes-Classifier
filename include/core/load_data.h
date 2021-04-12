#pragma once

#include <string>
#include <vector>
#include <core/image.h>

namespace naivebayes {
    using std::vector;
    using std::string;

    class LoadData {
    public:
        /**
         * LoadData constructor.
         * @param image_size image size of an image.
         * @param image_count number of images in a file.
         */
        LoadData(size_t image_size, size_t image_count);

        /**
         * Operator used to parse data.
         * @param input istream parameter.
         * @param data LoadData object passed as parameter.
         * @return input.
         */
        friend std::istream &operator>>(std::istream &input, LoadData &data);

        // getters
        const vector<Image> &GetImages() const;

        const size_t &GetImageCount() const;

        const size_t &GetImageSize() const;

        size_t GetNumberOfImagesInClass(int label) const;

    private:
        size_t image_size_;
        size_t image_count_;

        char kWhitePixel = ' ';
        char kGrayPixel = '+';
        char kBlackPixel = '#';

        int kWhitePixelValue = 0;
        int kGrayPixelValue = 1;
        int kBlackPixelValue = 2;

        vector<Image> num_of_images_per_label_;

        vector<int> num_of_images_in_class_;

        int kMagicTen = 10;

        /**
         * Helper method to fill image with pixels of different shades.
         * @param input istream parameter. 
         * @param data LoadData object passed as parameter.
         * @return 2d vector of pixels.i
         */
        vector<vector<int>> FillImageWithPixels(std::istream &input, LoadData &data);
    };

}  // namespace naivebayes
