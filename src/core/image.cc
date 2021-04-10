#include <utility>
#include <core/image.h>

namespace naivebayes {
    using std::vector;

    Image::Image(int label, size_t size, const vector<vector<int>> pixels) {
        label_ = label;
        size_ = size;
        pixels_ = pixels;
    }

    const size_t &Image::GetImageSize() const {
        return size_;
    }

    const int &Image::GetLabel() const {
        return label_;
    }

    const vector<vector<int>> &Image::GetPixels() const {
        return pixels_;
    }
}  // namespace naivebayes