#include <string>
#include <vector>
#include <core/image.h>

namespace naivebayes {
    using namespace std;

    class TrainingData {
    public:
        TrainingData(size_t image_size, size_t image_count);
        friend istream& operator >> (istream& input,TrainingData& data);

        vector<Image> getImages() const;
        // each class has a probability value
        // each pixel has a probability value
        // each P(class=c|all pixel values) has a probability value

    private:
        size_t image_size_;
        size_t image_count_;

        char kWhitePixel = ' ';

        vector<Image> images_;

    };

}  // namespace naivebayes
