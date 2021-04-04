#include <string>
#include <map>
#include <vector>
#include <core/image.h>

namespace naivebayes {
    using namespace std;

    class TrainingData {
    public:
        TrainingData(size_t image_size, size_t image_count);
        string GetBestClass() const;
        friend istream& operator >> (istream& input,TrainingData& data);

        // each class has a probability value
        // each pixel has a probability value
        // each P(class=c|all pixel values) has a probability value

    private:
        const static int kLaplaceSmoothing = 1;
        size_t image_size_;
        size_t image_count_;

        vector<Image> images_;

    };

}  // namespace naivebayes

/*
TODO: rename this file. You'll also need to modify CMakeLists.txt.

You can (and should) create more classes and files in include/core (header
 files) and src/core (source files); this project is too big to only have a
 single class.

Make sure to add any files that you create to CMakeLists.txt.

TODO Delete this comment before submitting your code.
*/
