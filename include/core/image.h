#include <string>
#include <map>
#include <vector>

namespace naivebayes {
    using namespace std;

    class Image {
    public:
        Image(int label, size_t size, vector<vector<bool>> pixels);

    private:
        vector<vector<bool>> pixels_;
        int label_;
        size_t size_;
    };

}  // namespace naivebayes
