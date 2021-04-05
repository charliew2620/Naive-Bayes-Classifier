#pragma once

#include <string>
#include <vector>

namespace naivebayes {
    using namespace std;

    class Image {
    public:
        Image(int label, size_t size, vector<vector<int>> pixels);

        size_t getImageSize() const;
        vector<vector<int>> getPixels() const;
        
        int GetLabel() const;

    private:
        vector<vector<int>> pixels_;
        int label_;
        size_t size_;
    };

}  // namespace naivebayes
