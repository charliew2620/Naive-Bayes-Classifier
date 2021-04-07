#pragma once

#include <string>
#include <vector>

namespace naivebayes {
    using namespace std;

    class Image {
    public:
        /**
         * Creates Image constructor.
         * @param label of the image anywhere from 0-9.
         * @param size of the image passed in.
         * @param pixels 2d vector of the pixels making up the image.
         */
        Image(int label, size_t size, vector<vector<int>> pixels);

        // getters
        const size_t &GetImageSize() const;

        const vector<vector<int>> &GetPixels() const;

        const int &GetLabel() const;

    private:
        vector<vector<int>> pixels_;
        int label_;
        size_t size_;
    };

}  // namespace naivebayes
