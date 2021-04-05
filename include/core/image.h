#pragma once

#include <string>
#include <vector>

namespace naivebayes {
    using namespace std;

    class Image {
    public:
        Image(int label, size_t size, vector<vector<bool>> pixels);

        size_t getImageSize() const;
        vector<vector<bool>> getPixels() const;
        
        int GetLabel() const;

    private:
        vector<vector<bool>> pixels_;
        int label_;
        size_t size_;
    };

}  // namespace naivebayes
