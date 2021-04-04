#include <core/image.h>
#include <core/model.h>


namespace naivebayes {
    using namespace  std;


    istream &operator>>(istream &input, Model &model) {
        return input;
    }
}  // namespace naivebayes