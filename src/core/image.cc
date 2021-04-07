#include <utility>
#include <core/image.h>

namespace naivebayes {
    using namespace std;

    Image::Image(int label, size_t size, vector<vector<int>> pixels) {
        label_ = label;
        size_ = size;
        pixels_ = std::move(pixels);
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