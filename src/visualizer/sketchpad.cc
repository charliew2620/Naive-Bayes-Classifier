#include <visualizer/sketchpad.h>

namespace naivebayes {

    namespace visualizer {

        using glm::vec2;

    void Sketchpad::ToggleCurrentBrushColor() {
            current_brush_color_ = current_brush_color_ == 1 ? 2: 1;
        }

        Sketchpad::Sketchpad(const vec2 &top_left_corner, size_t num_pixels_per_side,
                             double sketchpad_size, double brush_radius)
                : top_left_corner_(top_left_corner),
                  num_pixels_per_side_(num_pixels_per_side),
                  pixel_side_length_(sketchpad_size / num_pixels_per_side),
                  brush_radius_(brush_radius) {
            pixels_.resize(num_pixels_per_side_, std::vector<int>(num_pixels_per_side_));
        }

        void Sketchpad::Draw() const {
            for (size_t row = 0; row < num_pixels_per_side_; ++row) {
                for (size_t col = 0; col < num_pixels_per_side_; ++col) {
                    switch(pixels_[row][col]) {
                      case 2:
                        ci::gl::color(ci::Color::gray(0.3f));
                        break;
                      case 1:
                        ci::gl::color(ci::Color::gray(0.8f));
                        break;
                      default:
                        ci::gl::color(ci::Color("white"));
                    }

                    vec2 pixel_top_left = top_left_corner_ + vec2(col * pixel_side_length_,
                                                                  row * pixel_side_length_);

                    vec2 pixel_bottom_right =
                            pixel_top_left + vec2(pixel_side_length_, pixel_side_length_);
                    ci::Rectf pixel_bounding_box(pixel_top_left, pixel_bottom_right);

                    ci::gl::drawSolidRect(pixel_bounding_box);

                    ci::gl::color(ci::Color("black"));
                    ci::gl::drawStrokedRect(pixel_bounding_box);
                }
            }
        }

        void Sketchpad::HandleBrush(const vec2 &brush_screen_coords) {
            vec2 brush_sketchpad_coords =
                    (brush_screen_coords - top_left_corner_) / (float) pixel_side_length_;

            for (size_t row = 0; row < num_pixels_per_side_; ++row) {
                for (size_t col = 0; col < num_pixels_per_side_; ++col) {
                    vec2 pixel_center = {col + 0.5, row + 0.5};

                    if (glm::distance(brush_sketchpad_coords, pixel_center) <=
                        brush_radius_) {
                        pixels_[row][col] = current_brush_color_;
                    }
                }
            }
        }

        void Sketchpad::Clear() {
            pixels_.clear();
            pixels_.resize(num_pixels_per_side_, std::vector<int>(num_pixels_per_side_));
        }

        const std::vector<std::vector<int>> &Sketchpad::GetPixels() {
            return pixels_;
        }

    }  // namespace visualizer

}  // namespace naivebayes
