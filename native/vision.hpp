#pragma once

#include <opencv2/core.hpp>
#include <map>
#include <string>
#include <vector>

struct ShapeResult {
    const cv::Point* points;
    int count;
    double x, y, angle, area;
};

// Pointers into detector storage expire on the next observation.
struct Detection {
    int width, height, board_height, active, score;
    cv::Rect roi, player;
    const cv::Rect* obstacles;
    int obstacle_count;
    const ShapeResult* shapes;
    int shape_count;
    const char* ui;
    double ui_x, ui_y, ui_confidence;
    double presented_at, capture_seconds, detection_seconds;
};

class Detector {
public:
    explicit Detector(cv::Rect requested) : requested_(requested), roi_(requested) {}
    void add_template(const std::string& name, const cv::Mat& image);
    void inspect(const cv::Mat& frame, Detection& result);

private:
    void locate_board(const cv::Mat& frame);
    void find_player(Detection& result);
    void find_obstacles(Detection& result);
    int read_score();
    void inspect_ui(const cv::Mat& frame, Detection& result);
    cv::Rect requested_, roi_;
    cv::Size frame_size_;
    int frames_ = 0;
    cv::Mat board_, grey_, hsv_, dark_, solid_, scratch_, ui_image_, ui_grey_;
    const cv::Mat solid_kernel_ = cv::Mat::ones(9, 9, CV_8U);
    const cv::Mat score_kernel_ = cv::Mat::ones(23, 23, CV_8U);
    std::vector<cv::Rect> obstacles_, previous_bars_;
    std::vector<std::vector<cv::Point>> hulls_;
    std::vector<ShapeResult> shapes_;
    std::vector<std::pair<int, cv::Mat>> digits_;
    std::map<std::string, cv::Mat> templates_;
};
