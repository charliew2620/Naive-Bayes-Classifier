#include <utility>
#include <core/image.h>

namespace naivebayes {
    using namespace std;

    Image::Image(int label, size_t size, vector<vector<int>> pixels) {
        label_ = label;
        size_ = size;
        pixels_ = pixels;

    }

    size_t Image::getImageSize() const {
        return size_;
    }

    vector<vector<int>> Image::getPixels() const{
        return pixels_;
    }

    int Image::GetLabel() const {
        return label_;
    }
}  // namespace naivebayes