#include <string>
#include <map>
#include <vector>

namespace naivebayes {
    using namespace std;

    class TrainingData {
    public:
        string GetBestClass() const;

        // each class has a probability value
        // each pixel has a probability value
        // each P(class=c|all pixel values) has a probability value

    private:
        const static int kLaplaceSmoothing = 1;

        map<int, double> calculate_class_probability(map<int, vector<vector<float>>> data, float training_label);

        // result of this method is then passed into class probability method to calculate?
        map<int, vector<vector<double>>> calculate_image_probability(vector<vector<float>> data, float training_label);


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
