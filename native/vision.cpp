#include "vision.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace {
constexpr int BoardWidth = 420;
using Contours = std::vector<std::vector<cv::Point>>;

void to_grey(const cv::Mat& color, cv::Mat& grey) {
    cv::cvtColor(color, grey, color.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
}

Contours contours(const cv::Mat& mask) {
    Contours result;
    cv::findContours(mask, result, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return result;
}

cv::Mat normalized_digit(const cv::Mat& glyph) {
    cv::Mat resized, result = cv::Mat::zeros(56, 54, CV_32F);
    cv::resize(glyph, resized, {std::max(1, cvRound(glyph.cols * (52.0 / glyph.rows))), 52});
    const int width = std::min(50, resized.cols);
    resized(cv::Rect(0, 0, width, 52)).convertTo(
        result(cv::Rect((54 - width) / 2, 2, width, 52)), CV_32F, 1.0 / 255);
    return result;
}

bool contained(const cv::Rect& box, const cv::Rect& player, int top_tolerance) {
    return !player.empty() && box.x >= player.x - 1 && box.br().x <= player.br().x + 1
        && box.y >= player.y - top_tolerance && box.br().y <= player.br().y + 1;
}

int median(std::vector<int>& values) {
    if (values.empty()) return 0;
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return values.size() % 2 ? *middle : (*middle + *std::max_element(values.begin(), middle)) / 2;
}
} // namespace

void Detector::add_template(const std::string& name, const cv::Mat& image) {
    if (name.size() == 1 && name[0] >= '0' && name[0] <= '9') {
        digits_.emplace_back(name[0] - '0', normalized_digit(image));
        std::stable_sort(digits_.begin(), digits_.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
    } else {
        templates_[name] = image.clone();
    }
}

void Detector::locate_board(const cv::Mat& frame) {
    const double scale = std::min(1.0, double(BoardWidth) / frame.cols);
    cv::Mat small, grey, flat, mask, labels, stats, centers;
    cv::resize(frame, small, {}, scale, scale);
    to_grey(small, grey);
    flat.create(grey.size(), CV_8U);
    std::array<int, 256> histogram{};
    int eligible = 0;
    for (int y = 0; y < small.rows; ++y) {
        const auto* pixels = small.ptr<unsigned char>(y);
        const auto* values = grey.ptr<unsigned char>(y);
        auto* row = flat.ptr<unsigned char>(y);
        for (int x = 0; x < small.cols; ++x) {
            const auto* p = pixels + x * small.channels();
            row[x] = std::max({p[0], p[1], p[2]}) - std::min({p[0], p[1], p[2]}) < 12;
            const int value = values[x];
            if (row[x] && 170 < value && value < 245) {
                ++histogram[value];
                ++eligible;
            }
        }
    }
    if (eligible < 1000) return;
    const int mode = int(std::max_element(histogram.begin(), histogram.end()) - histogram.begin());
    cv::inRange(grey, mode - 7, mode + 7, mask);
    cv::multiply(mask, flat, mask);
    int count = cv::connectedComponentsWithStats(mask, labels, stats, centers);
    if (count < 2) return;
    int biggest = 1;
    for (int i = 2; i < count; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = i;
    }
    const int* s = stats.ptr<int>(biggest);
    cv::Rect component(s[0], s[1], s[2], s[3]);
    if (s[4] < small.total() * 0.2 || component.height < component.width) return;
    const cv::Mat crop = mask(component);
    std::vector<int> coverage;
    int eligible_rows = 0;
    for (int y = 0; y < crop.rows; ++y) {
        coverage.push_back(cv::countNonZero(crop.row(y)));
        eligible_rows += coverage.back() > 0.65 * crop.cols;
    }
    if (eligible_rows < crop.rows * 0.35) return;
    int top = -1;
    for (int y = 0; y + 8 < crop.rows; ++y) {
        if (coverage[y] > 0.65 * crop.cols
            && cv::countNonZero(crop.rowRange(y, y + 8)) > 0.65 * crop.cols * 8) {
            top = y;
            break;
        }
    }
    if (top < 0) return;
    int first_white = -1, last_white = -1, white_rows = 0;
    for (int y = 0; y < std::min(grey.rows / 5, 130); ++y) {
        cv::Mat row = grey(cv::Rect(component.x, y, component.width, 1));
        if (cv::countNonZero(row > 247) > 0.55 * row.cols) {
            if (first_white < 0) first_white = y;
            last_white = y;
            ++white_rows;
        }
    }
    if (white_rows >= 8 && first_white < component.y + top) {
        top = std::max(0, last_white + 2 - component.y);
    }
    if (top >= crop.rows) return;
    const cv::Mat body = crop.rowRange(top, crop.rows);
    int left = 0, right = 0, usable_columns = 0;
    for (int x = 0; x < body.cols; ++x) {
        if (cv::countNonZero(body.col(x)) > 0.45 * body.rows) {
            if (!usable_columns) left = x;
            right = x + 1;
            ++usable_columns;
        }
    }
    if (usable_columns < crop.cols * 0.6) return;
    const int x = cvRound((component.x + left) / scale);
    const int y = cvRound((component.y + top) / scale);
    const int x1 = std::min(frame.cols, cvRound((component.x + right) / scale));
    const int y1 = std::min(frame.rows, cvRound(component.br().y / scale));
    roi_ = {x, y, x1 - x, y1 - y};
}

void Detector::find_player(Detection& result) {
    cv::inRange(hsv_, cv::Scalar(0, 71, 101), cv::Scalar(179, 255, 255), scratch_);
    double best = -1.0;
    for (const auto& band : contours(scratch_)) {
        const auto r = cv::boundingRect(band);
        if (r.width <= 8 || r.width >= 55 || r.height < 2 || r.height >= 28 || r.y <= 90) continue;
        const int x0 = std::max(0, r.x - 3), y0 = std::max(0, r.y - 6);
        const int x1 = std::min(dark_.cols, r.br().x + 8), y1 = std::min(dark_.rows, r.y + 42);
        const auto bodies = contours(dark_(cv::Rect(x0, y0, x1 - x0, y1 - y0)));
        for (const auto& body : bodies) {
            const auto b = cv::boundingRect(body);
            if (b.width <= 12 || b.width >= 42 || b.height < 8 || b.height >= 42
                || std::abs(x0 + b.x + b.width / 2.0 - (r.x + r.width / 2.0)) >= 9
                || y0 + b.y > r.br().y + 8 || y0 + b.br().y < r.br().y - 3) continue;
            const double area = cv::contourArea(body);
            if (area <= best) continue;
            int top = std::min(y0 + b.y, r.y - 3);
            for (const auto& cap : bodies) {
                const auto c = cv::boundingRect(cap);
                if (std::abs(c.x - b.x) <= 3 && std::abs(c.width - b.width) <= 3
                    && r.y - 10 <= y0 + c.y && y0 + c.y < r.y) top = std::min(top, y0 + c.y);
            }
            result.player = {x0 + b.x, top, b.width, y0 + b.br().y - top};
            best = area;
        }
    }
}

void Detector::find_obstacles(Detection& result) {
    const int badge_height = std::min(110, hsv_.rows);
    cv::inRange(hsv_(cv::Rect(0, 0, 160, badge_height)), cv::Scalar(0, 46, 76),
                cv::Scalar(179, 255, 255), scratch_);
    int badge_top = 0, badge_bottom = 0, badge_right = 0;
    if (cv::countNonZero(scratch_) > 15) {
        const auto colored = cv::boundingRect(scratch_);
        badge_top = std::max(0, colored.y - 28);
        badge_bottom = std::min(badge_height, colored.br().y + 9);
        badge_right = std::min(160, std::max(135, colored.br().x + 9));
    }
    for (bool left : {true, false}) {
        const int edge = left ? 2 : BoardWidth - 3;
        for (int top = 0; top < dark_.rows;) {
            if (!dark_.ptr<unsigned char>(top)[edge]) {
                ++top;
                continue;
            }
            int bottom = top + 1;
            while (bottom < dark_.rows && dark_.ptr<unsigned char>(bottom)[edge]) ++bottom;
            if (6 <= bottom - top && bottom - top <= 250) {
                std::vector<int> lengths, visible;
                lengths.reserve(bottom - top);
                for (int y = top; y < bottom; ++y) {
                    const auto* row = dark_.ptr<unsigned char>(y);
                    const bool occluded = left && badge_top <= y && y < badge_bottom;
                    int x = occluded ? badge_right : 2;
                    while (x < BoardWidth && row[left ? x : BoardWidth - 1 - x]) ++x;
                    const int extent = x == BoardWidth ? 2 : x;
                    lengths.push_back(extent);
                    if (!occluded) visible.push_back(extent);
                }
                int extent = median(lengths);
                if (left && visible.size() >= 2) extent = median(visible);
                if (left && visible.size() < 2 && badge_top < bottom && top < badge_bottom) {
                    int distance = 65;
                    for (const auto& b : previous_bars_) {
                        if (b.x < 4 && std::abs(b.y - top) < distance
                            && std::abs(b.height - (bottom - top)) < 25) {
                            extent = b.width + 1;
                            distance = std::abs(b.y - top);
                        }
                    }
                }
                if (extent >= 8) {
                    cv::Rect b(left ? 1 : BoardWidth - extent, top, extent - 1, bottom - top);
                    if (!contained(b, result.player, 1)) obstacles_.push_back(b);
                }
            }
            top = bottom;
        }
    }
    previous_bars_.clear();
    if (result.active) previous_bars_ = obstacles_;

    cv::morphologyEx(dark_, solid_, cv::MORPH_OPEN, solid_kernel_);
    for (const auto& contour : contours(solid_)) {
        const auto b = cv::boundingRect(contour);
        if (b.height < 6 || b.height > 250 || b.width < 6 || b.x < 4 || b.br().x > 416
            || contained(b, result.player, 0)) continue;
        const double area = cv::contourArea(contour);
        const double fill = area / b.area();
        const double ratio = double(b.width) / b.height;
        const bool occluded_square = b.y < 130 && b.width >= 50 && 0.8 < ratio && ratio < 1.25 && fill > 0.65;
        if (fill < 0.78 && !occluded_square) {
            if (area > 450 && b.width > 40 && b.height > 40) {
                const auto m = cv::moments(contour);
                if (m.m00) {
                    auto& hull = hulls_.emplace_back();
                    cv::convexHull(contour, hull);
                    shapes_.push_back({hull.data(), int(hull.size()), m.m10 / m.m00, m.m01 / m.m00,
                        0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02), m.m00});
                }
            }
            continue;
        }
        if (b.y < 130 && fill < 0.90 && !occluded_square) continue;
        if (ratio >= 0.7) obstacles_.push_back(b);
    }
    cv::morphologyEx(dark_.rowRange(0, std::min(dark_.rows, 180)), solid_, cv::MORPH_OPEN, score_kernel_);
    for (const auto& contour : contours(solid_)) {
        const auto b = cv::boundingRect(contour);
        if (b.y >= 130 || b.width < 25 || b.width > 36 || b.height < 25 || b.height > 36
            || double(b.width) / b.height <= 0.85 || double(b.width) / b.height >= 1.18
            || cv::contourArea(contour) / b.area() <= 0.85) continue;
        if (std::none_of(obstacles_.begin(), obstacles_.end(), [&b](const cv::Rect& a) {
            return std::abs(a.x - b.x) < 3 && std::abs(a.y - b.y) < 3
                && std::abs(a.width - b.width) < 4 && std::abs(a.height - b.height) < 4;
        })) obstacles_.push_back(b);
    }
    result.obstacles = obstacles_.data();
    result.obstacle_count = int(obstacles_.size());
    result.shapes = shapes_.data();
    result.shape_count = int(shapes_.size());
}

int Detector::read_score() {
    if (dark_.rows <= 35) return -1;
    const cv::Mat mask = dark_.rowRange(35, std::min(dark_.rows, 135));
    std::vector<cv::Rect> boxes;
    for (const auto& contour : contours(mask)) {
        const auto b = cv::boundingRect(contour);
        if (40 <= b.height && b.height <= 72 && 3 < b.width && b.width < 60 && 80 < b.x && b.x < 340)
            boxes.push_back(b);
    }
    if (boxes.empty() || boxes.size() > 4) return -1;
    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) {
        return std::tie(a.x, a.y, a.width, a.height) < std::tie(b.x, b.y, b.width, b.height);
    });
    auto [top, bottom] = std::minmax_element(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) {
        return a.y < b.y;
    });
    if (bottom->y - top->y > 4) return -1;
    int score = 0;
    for (auto box : boxes) {
        const cv::Mat digit = normalized_digit(mask(box));
        double best = 0.11 * digit.total();
        int best_digit = -1;
        for (const auto& [number, example] : digits_) {
            const double distance = cv::norm(digit, example, cv::NORM_L2SQR);
            if (distance < best) {
                best = distance;
                best_digit = number;
            }
        }
        if (best_digit < 0) return -1;
        score = score * 10 + best_digit;
    }
    return score;
}

void Detector::inspect_ui(const cv::Mat& frame, Detection& result) {
    const double factor = double(BoardWidth) / frame.cols;
    cv::resize(frame, ui_image_, {BoardWidth, cvRound(frame.rows * factor)});
    to_grey(ui_image_, ui_grey_);
    struct Match { double confidence = 0; cv::Point p; cv::Size size; };
    const auto match = [&](const char* name) {
        Match found;
        const auto entry = templates_.find(name);
        if (entry == templates_.end()) return found;
        const auto& glyph = entry->second;
        if (ui_grey_.rows < glyph.rows || ui_grey_.cols < glyph.cols) return found;
        cv::matchTemplate(ui_grey_, glyph, scratch_, cv::TM_CCOEFF_NORMED);
        cv::minMaxLoc(scratch_, nullptr, &found.confidence, nullptr, &found.p);
        found.size = glyph.size();
        return found;
    };
    const auto action = [&](const char* name, const Match& button, double confidence) {
        result.ui = name;
        result.ui_x = (button.p.x + button.size.width / 2.0) / factor;
        result.ui_y = (button.p.y + button.size.height / 2.0) / factor;
        result.ui_confidence = confidence;
    };
    const auto outfit = match("outfit_label"), use = match("outfit_use");
    if (outfit.confidence > 0.9 && use.confidence > 0.9 && use.p.x > 220 && use.p.y > outfit.p.y + 150) {
        action("use_outfit", use, std::min(outfit.confidence, use.confidence));
        return;
    }
    const auto ad = match("ad_close");
    if (ad.confidence > 0.91 && ad.p.x > 280 && 150 < ad.p.y && ad.p.y < 420) {
        const int top = std::min(ui_grey_.rows - 1, ad.p.y + 200);
        const int bottom = std::min(ui_grey_.rows, ad.p.y + 400);
        const auto card = ui_grey_(cv::Rect(50, top, 320, bottom - top));
        if (cv::countNonZero(card > 235) > 0.55 * card.total()) {
            action("close_ad", ad, ad.confidence);
            return;
        }
    }
    const auto score = match("score"), restart = match("restart");
    if (score.confidence > 0.88 && restart.confidence > 0.88
        && score.p.y < ui_grey_.rows * 0.5 && restart.p.y > ui_grey_.rows * 0.5) {
        action("restart", restart, std::min(score.confidence, restart.confidence));
        return;
    }
    const auto cont = match("continue_label"), close = match("continue_close");
    const auto gap = close.p - cont.p;
    if (cont.confidence > 0.88 && close.confidence > 0.88
        && 90 < gap.x && gap.x < 190 && -70 < gap.y && gap.y < 10) {
        action("decline_continue", close, std::min(cont.confidence, close.confidence));
        return;
    }
    const auto helmet = match("helmet_label"), decline = match("helmet_no");
    if (helmet.confidence > 0.88 && decline.confidence > 0.88 && decline.p.y > helmet.p.y)
        action("decline_helmet", decline, std::min(helmet.confidence, decline.confidence));
}

void Detector::inspect(const cv::Mat& frame, Detection& result) {
    result = {};
    result.width = frame.cols;
    result.height = frame.rows;
    result.score = -1;
    obstacles_.clear();
    shapes_.clear();
    hulls_.clear();
    if (frame_size_ != frame.size()) {
        roi_ = requested_;
        frames_ = 0;
        previous_bars_.clear();
        frame_size_ = frame.size();
    }
    if (requested_.empty() && roi_.empty() && frames_ % 6 == 0) locate_board(frame);
    ++frames_;
    if (!roi_.empty() && (roi_ & cv::Rect(0, 0, frame.cols, frame.rows)) == roi_) {
        const int height = cvRound(roi_.height * double(BoardWidth) / roi_.width);
        if (height > 0) {
            result.roi = roi_;
            result.board_height = height;
            cv::Mat board = frame(roi_);
            if (roi_.width != BoardWidth) {
                cv::resize(board, board_, {BoardWidth, height});
                board = board_;
            }
            to_grey(board, grey_);
            cv::threshold(grey_, dark_, 77, 255, cv::THRESH_BINARY_INV);
            cv::threshold(grey_, scratch_, 170, 255, cv::THRESH_BINARY);
            const int light = cv::countNonZero(scratch_);
            result.active = unsigned(2 * light) > grey_.total();
            if (unsigned(2 * light) == grey_.total()) {
                // The two middle greys can straddle the threshold.
                double below = 0, above = 255;
                cv::minMaxLoc(grey_, nullptr, &below, nullptr, nullptr, ~scratch_);
                cv::minMaxLoc(grey_, &above, nullptr, nullptr, nullptr, scratch_);
                result.active = (below + above) / 2 > 170;
            }
            cv::cvtColor(board, hsv_, cv::COLOR_BGR2HSV);
            find_player(result);
            find_obstacles(result);
            result.score = read_score();
        }
    }
    if (!result.active || result.player.empty()) inspect_ui(frame, result);
}
