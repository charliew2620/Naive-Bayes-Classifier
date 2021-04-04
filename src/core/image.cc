#include <core/image.h>

#include <utility>

namespace naivebayes {
    using namespace  std;

    Image::Image(int label, size_t size, vector<vector<bool>> pixels) {
        label_ = label;
        size_ = size;
        pixels_ = std::move(pixel);

    }


}  // namespace naivebayes