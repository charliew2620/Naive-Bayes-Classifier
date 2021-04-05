#pragma once

#include <string>
#include <vector>

namespace naivebayes {
    using namespace std;

    class Image {
    public:
        Image(int label, size_t size, vector<vector<int>> pixels);

        const size_t &GetImageSize() const;

        const vector<vector<int>> &GetPixels() const;

        const int &GetLabel() const;

    private:
        vector<vector<int>> pixels_;
        int label_;
        size_t size_;
    };

}  // namespace naivebayes
