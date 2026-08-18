#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <net.h>

namespace {

struct Options {
    std::string mode;
    std::string model_dir;
    std::string image_list;
    std::string predictions_csv;
    std::string images_csv;
    std::string summary_csv;

    int imgsz = 640;
    int threads = 2;
    int warmup = 5;
    int repeat = 3;
    int max_det = 300;
    int limit = 0;

    float conf = 0.25F;
    float iou = 0.70F;
};

struct Metadata {
    std::vector<std::string> class_names;
};

struct LetterboxInfo {
    float scale = 1.0F;
    int pad_left = 0;
    int pad_top = 0;
    int original_width = 0;
    int original_height = 0;
};

struct Detection {
    int class_id = -1;
    float confidence = 0.0F;
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
};

struct Timing {
    double io_ms = 0.0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;

    double compute_ms() const {
        return preprocess_ms + inference_ms + postprocess_ms;
    }

    double total_ms() const {
        return io_ms + compute_ms();
    }
};

struct OutputLayout {
    enum class Kind {
        FEATURE_ROWS,
        ANCHOR_ROWS
    };

    Kind kind = Kind::FEATURE_ROWS;
    int anchors = 0;
    int features = 0;
};

struct InferenceResult {
    std::vector<Detection> detections;
    Timing timing;
    int output_dims = 0;
    int output_w = 0;
    int output_h = 0;
    int output_c = 0;
};

std::string trim(const std::string& value) {
    const std::string whitespace = " \t\r\n";
    const std::size_t first = value.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

bool file_exists(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode);
}

std::string join_path(const std::string& left, const std::string& right) {
    if (left.empty()) {
        return right;
    }
    if (left.back() == '/') {
        return left + right;
    }
    return left + "/" + right;
}

std::string base_name(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string strip_extension(const std::string& filename) {
    const std::size_t dot = filename.find_last_of('.');
    return dot == std::string::npos ? filename : filename.substr(0, dot);
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (char character : value) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

void print_usage() {
    std::cerr
        << "Usage:\n"
        << "  Validate:\n"
        << "    yolo26_ncnn_runner --mode validate --model-dir DIR "
        << "--image-list FILE --predictions FILE --images-summary FILE "
        << "[--imgsz 640] [--threads 2] [--conf 0.001] "
        << "[--iou 0.70] [--max-det 300] [--limit N]\n\n"
        << "  Benchmark:\n"
        << "    yolo26_ncnn_runner --mode benchmark --model-dir DIR "
        << "--image-list FILE --summary FILE "
        << "[--imgsz 640] [--threads 2] [--warmup 5] "
        << "[--repeat 3] [--conf 0.25] [--iou 0.70] "
        << "[--max-det 300] [--limit N]\n";
}

std::string require_value(int argc, char* argv[], int& index) {
    if (index + 1 >= argc) {
        throw std::runtime_error(
            std::string("Missing value after argument: ") + argv[index]
        );
    }
    ++index;
    return argv[index];
}

int parse_int(const std::string& text, const std::string& name) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error("Invalid integer for " + name + ": " + text);
    }
    return value;
}

float parse_float(const std::string& text, const std::string& name) {
    std::size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error("Invalid number for " + name + ": " + text);
    }
    return value;
}

Options parse_options(int argc, char* argv[]) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--help" || argument == "-h") {
            print_usage();
            std::exit(0);
        } else if (argument == "--mode") {
            options.mode = require_value(argc, argv, index);
        } else if (argument == "--model-dir") {
            options.model_dir = require_value(argc, argv, index);
        } else if (argument == "--image-list") {
            options.image_list = require_value(argc, argv, index);
        } else if (argument == "--predictions") {
            options.predictions_csv = require_value(argc, argv, index);
        } else if (argument == "--images-summary") {
            options.images_csv = require_value(argc, argv, index);
        } else if (argument == "--summary") {
            options.summary_csv = require_value(argc, argv, index);
        } else if (argument == "--imgsz") {
            options.imgsz = parse_int(
                require_value(argc, argv, index), "--imgsz"
            );
        } else if (argument == "--threads") {
            options.threads = parse_int(
                require_value(argc, argv, index), "--threads"
            );
        } else if (argument == "--warmup") {
            options.warmup = parse_int(
                require_value(argc, argv, index), "--warmup"
            );
        } else if (argument == "--repeat") {
            options.repeat = parse_int(
                require_value(argc, argv, index), "--repeat"
            );
        } else if (argument == "--max-det") {
            options.max_det = parse_int(
                require_value(argc, argv, index), "--max-det"
            );
        } else if (argument == "--limit") {
            options.limit = parse_int(
                require_value(argc, argv, index), "--limit"
            );
        } else if (argument == "--conf") {
            options.conf = parse_float(
                require_value(argc, argv, index), "--conf"
            );
        } else if (argument == "--iou") {
            options.iou = parse_float(
                require_value(argc, argv, index), "--iou"
            );
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    if (options.mode != "validate" && options.mode != "benchmark") {
        throw std::runtime_error(
            "--mode must be either validate or benchmark."
        );
    }
    if (options.model_dir.empty()) {
        throw std::runtime_error("--model-dir is required.");
    }
    if (options.image_list.empty()) {
        throw std::runtime_error("--image-list is required.");
    }
    if (options.imgsz <= 0) {
        throw std::runtime_error("--imgsz must be greater than zero.");
    }
    if (options.threads <= 0) {
        throw std::runtime_error("--threads must be greater than zero.");
    }
    if (options.conf < 0.0F || options.conf > 1.0F) {
        throw std::runtime_error("--conf must be in [0, 1].");
    }
    if (options.iou <= 0.0F || options.iou > 1.0F) {
        throw std::runtime_error("--iou must be in (0, 1].");
    }
    if (options.max_det <= 0) {
        throw std::runtime_error("--max-det must be greater than zero.");
    }
    if (options.limit < 0) {
        throw std::runtime_error("--limit cannot be negative.");
    }

    if (options.mode == "validate") {
        if (options.predictions_csv.empty()) {
            throw std::runtime_error(
                "--predictions is required in validate mode."
            );
        }
        if (options.images_csv.empty()) {
            throw std::runtime_error(
                "--images-summary is required in validate mode."
            );
        }
    } else {
        if (options.summary_csv.empty()) {
            throw std::runtime_error(
                "--summary is required in benchmark mode."
            );
        }
        if (options.warmup < 0) {
            throw std::runtime_error("--warmup cannot be negative.");
        }
        if (options.repeat <= 0) {
            throw std::runtime_error("--repeat must be greater than zero.");
        }
    }

    return options;
}

std::vector<std::string> read_image_list(
    const std::string& path,
    int limit
) {
    if (!file_exists(path)) {
        throw std::runtime_error("Image list not found: " + path);
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open image list: " + path);
    }

    std::vector<std::string> images;
    std::string line;

    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!file_exists(line)) {
            throw std::runtime_error(
                "Image listed but not found: " + line
            );
        }
        images.push_back(line);

        if (limit > 0 && static_cast<int>(images.size()) >= limit) {
            break;
        }
    }

    if (images.empty()) {
        throw std::runtime_error(
            "The image list does not contain any usable images: " + path
        );
    }

    return images;
}

std::string remove_matching_quotes(const std::string& value) {
    if (value.size() >= 2) {
        const char first = value.front();
        const char last = value.back();
        if ((first == '\'' && last == '\'') ||
            (first == '"' && last == '"')) {
            return value.substr(1, value.size() - 2);
        }
    }
    return value;
}

Metadata read_metadata(const std::string& metadata_path) {
    if (!file_exists(metadata_path)) {
        throw std::runtime_error(
            "Metadata file not found: " + metadata_path
        );
    }

    std::ifstream input(metadata_path);
    if (!input) {
        throw std::runtime_error(
            "Unable to open metadata file: " + metadata_path
        );
    }

    Metadata metadata;
    bool in_names = false;
    std::map<int, std::string> indexed_names;
    std::string line;

    while (std::getline(input, line)) {
        const std::string stripped = trim(line);

        if (stripped == "names:") {
            in_names = true;
            continue;
        }

        if (!in_names) {
            continue;
        }

        // A non-indented YAML key marks the end of the names mapping.
        if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
            break;
        }

        if (stripped.empty()) {
            continue;
        }

        const std::size_t colon = stripped.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        const std::string key_text = trim(stripped.substr(0, colon));
        const std::string value_text = remove_matching_quotes(
            trim(stripped.substr(colon + 1))
        );

        try {
            const int class_id = parse_int(key_text, "metadata class id");
            indexed_names[class_id] = value_text;
        } catch (const std::exception&) {
            throw std::runtime_error(
                "Invalid class entry in metadata: " + stripped
            );
        }
    }

    if (indexed_names.empty()) {
        throw std::runtime_error(
            "No class names were found in metadata: " + metadata_path
        );
    }

    const int highest_id = indexed_names.rbegin()->first;
    metadata.class_names.resize(
        static_cast<std::size_t>(highest_id + 1)
    );

    for (int class_id = 0; class_id <= highest_id; ++class_id) {
        const auto found = indexed_names.find(class_id);
        if (found == indexed_names.end()) {
            throw std::runtime_error(
                "Metadata class ids must be continuous from zero."
            );
        }
        metadata.class_names[static_cast<std::size_t>(class_id)] =
            found->second;
    }

    return metadata;
}

ncnn::Mat preprocess_image(
    const cv::Mat& original,
    int imgsz,
    LetterboxInfo& info
) {
    if (original.empty()) {
        throw std::runtime_error("Cannot preprocess an empty image.");
    }

    info.original_width = original.cols;
    info.original_height = original.rows;
    info.scale = std::min(
        static_cast<float>(imgsz) /
            static_cast<float>(info.original_width),
        static_cast<float>(imgsz) /
            static_cast<float>(info.original_height)
    );

    const int resized_width = static_cast<int>(
        std::round(
            static_cast<float>(info.original_width) * info.scale
        )
    );
    const int resized_height = static_cast<int>(
        std::round(
            static_cast<float>(info.original_height) * info.scale
        )
    );

    const int horizontal_padding = imgsz - resized_width;
    const int vertical_padding = imgsz - resized_height;

    // Ultralytics letterbox uses centered padding. The +/-0.1 rounding keeps
    // the two sides consistent when the total padding is odd.
    info.pad_left = static_cast<int>(
        std::round(horizontal_padding / 2.0F - 0.1F)
    );
    const int pad_right = static_cast<int>(
        std::round(horizontal_padding / 2.0F + 0.1F)
    );
    info.pad_top = static_cast<int>(
        std::round(vertical_padding / 2.0F - 0.1F)
    );
    const int pad_bottom = static_cast<int>(
        std::round(vertical_padding / 2.0F + 0.1F)
    );

    cv::Mat resized;
    cv::resize(
        original,
        resized,
        cv::Size(resized_width, resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    cv::Mat letterboxed;
    cv::copyMakeBorder(
        resized,
        letterboxed,
        info.pad_top,
        pad_bottom,
        info.pad_left,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    if (letterboxed.cols != imgsz || letterboxed.rows != imgsz) {
        throw std::runtime_error(
            "Letterbox preprocessing produced an unexpected image size."
        );
    }

    if (!letterboxed.isContinuous()) {
        letterboxed = letterboxed.clone();
    }

    // OpenCV images are BGR. YOLO was trained with RGB tensors, so NCNN
    // performs BGR-to-RGB conversion while copying the pixels.
    ncnn::Mat input = ncnn::Mat::from_pixels(
        letterboxed.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        imgsz,
        imgsz
    );

    // Ultralytics input tensors use float values in [0, 1].
    const float normalization[3] = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F
    };
    input.substract_mean_normalize(nullptr, normalization);

    return input;
}

OutputLayout determine_output_layout(
    const ncnn::Mat& output,
    int class_count
) {
    const int expected_features = class_count + 4;

    if (output.elempack != 1) {
        throw std::runtime_error(
            "Unsupported packed output. Expected elempack=1, received " +
            std::to_string(output.elempack)
        );
    }

    if (output.dims != 2) {
        std::ostringstream message;
        message
            << "Unsupported NCNN output dimensions. Expected a 2D tensor, "
            << "received dims=" << output.dims
            << " w=" << output.w
            << " h=" << output.h
            << " c=" << output.c;
        throw std::runtime_error(message.str());
    }

    OutputLayout layout;

    if (output.h == expected_features) {
        layout.kind = OutputLayout::Kind::FEATURE_ROWS;
        layout.features = output.h;
        layout.anchors = output.w;
        return layout;
    }

    if (output.w == expected_features) {
        layout.kind = OutputLayout::Kind::ANCHOR_ROWS;
        layout.features = output.w;
        layout.anchors = output.h;
        return layout;
    }

    std::ostringstream message;
    message
        << "Output tensor does not match metadata class count. "
        << "Classes=" << class_count
        << ", expected features=" << expected_features
        << ", output w=" << output.w
        << ", output h=" << output.h;
    throw std::runtime_error(message.str());
}

float output_value(
    const ncnn::Mat& output,
    const OutputLayout& layout,
    int feature,
    int anchor
) {
    if (layout.kind == OutputLayout::Kind::FEATURE_ROWS) {
        return output.row(feature)[anchor];
    }
    return output.row(anchor)[feature];
}

float intersection_over_union(
    const Detection& left,
    const Detection& right
) {
    const float intersection_left = std::max(left.x1, right.x1);
    const float intersection_top = std::max(left.y1, right.y1);
    const float intersection_right = std::min(left.x2, right.x2);
    const float intersection_bottom = std::min(left.y2, right.y2);

    const float intersection_width =
        std::max(0.0F, intersection_right - intersection_left);
    const float intersection_height =
        std::max(0.0F, intersection_bottom - intersection_top);
    const float intersection_area =
        intersection_width * intersection_height;

    const float left_area =
        std::max(0.0F, left.x2 - left.x1) *
        std::max(0.0F, left.y2 - left.y1);
    const float right_area =
        std::max(0.0F, right.x2 - right.x1) *
        std::max(0.0F, right.y2 - right.y1);

    const float union_area =
        left_area + right_area - intersection_area;

    return union_area <= 0.0F
        ? 0.0F
        : intersection_area / union_area;
}

std::vector<Detection> class_aware_nms(
    const std::vector<Detection>& proposals,
    int class_count,
    float iou_threshold,
    int max_det
) {
    std::vector<std::vector<Detection>> grouped(
        static_cast<std::size_t>(class_count)
    );

    for (const Detection& detection : proposals) {
        grouped[static_cast<std::size_t>(detection.class_id)].push_back(
            detection
        );
    }

    std::vector<Detection> kept;

    for (std::vector<Detection>& class_detections : grouped) {
        std::sort(
            class_detections.begin(),
            class_detections.end(),
            [](const Detection& left, const Detection& right) {
                return left.confidence > right.confidence;
            }
        );

        std::vector<unsigned char> suppressed(
            class_detections.size(),
            0
        );

        for (std::size_t index = 0;
             index < class_detections.size();
             ++index) {
            if (suppressed[index] != 0) {
                continue;
            }

            kept.push_back(class_detections[index]);

            for (std::size_t other = index + 1;
                 other < class_detections.size();
                 ++other) {
                if (suppressed[other] != 0) {
                    continue;
                }

                if (intersection_over_union(
                        class_detections[index],
                        class_detections[other]
                    ) > iou_threshold) {
                    suppressed[other] = 1;
                }
            }
        }
    }

    std::sort(
        kept.begin(),
        kept.end(),
        [](const Detection& left, const Detection& right) {
            return left.confidence > right.confidence;
        }
    );

    if (static_cast<int>(kept.size()) > max_det) {
        kept.resize(static_cast<std::size_t>(max_det));
    }

    return kept;
}

std::vector<Detection> decode_output(
    const ncnn::Mat& output,
    const OutputLayout& layout,
    const LetterboxInfo& letterbox,
    int class_count,
    float confidence_threshold,
    float iou_threshold,
    int max_det
) {
    std::vector<Detection> proposals;
    proposals.reserve(static_cast<std::size_t>(layout.anchors));

    for (int anchor = 0; anchor < layout.anchors; ++anchor) {
        int best_class = -1;
        float best_confidence = -std::numeric_limits<float>::infinity();

        for (int class_id = 0; class_id < class_count; ++class_id) {
            const float score = output_value(
                output,
                layout,
                class_id + 4,
                anchor
            );
            if (score > best_confidence) {
                best_confidence = score;
                best_class = class_id;
            }
        }

        if (best_confidence < confidence_threshold) {
            continue;
        }

        // The exported end2end=false YOLO26 graph already applies DFL box
        // decoding, anchor points, stride multiplication, and class sigmoid.
        // Therefore out0 contains xywh boxes in 640x640 input coordinates
        // followed by per-class probabilities.
        const float center_x = output_value(output, layout, 0, anchor);
        const float center_y = output_value(output, layout, 1, anchor);
        const float width = output_value(output, layout, 2, anchor);
        const float height = output_value(output, layout, 3, anchor);

        float x1 = center_x - width / 2.0F;
        float y1 = center_y - height / 2.0F;
        float x2 = center_x + width / 2.0F;
        float y2 = center_y + height / 2.0F;

        // Remove letterbox padding and return coordinates in the original
        // image space, which is required for correct AP evaluation.
        x1 = (x1 - static_cast<float>(letterbox.pad_left)) /
             letterbox.scale;
        y1 = (y1 - static_cast<float>(letterbox.pad_top)) /
             letterbox.scale;
        x2 = (x2 - static_cast<float>(letterbox.pad_left)) /
             letterbox.scale;
        y2 = (y2 - static_cast<float>(letterbox.pad_top)) /
             letterbox.scale;

        x1 = std::max(
            0.0F,
            std::min(
                x1,
                static_cast<float>(letterbox.original_width)
            )
        );
        y1 = std::max(
            0.0F,
            std::min(
                y1,
                static_cast<float>(letterbox.original_height)
            )
        );
        x2 = std::max(
            0.0F,
            std::min(
                x2,
                static_cast<float>(letterbox.original_width)
            )
        );
        y2 = std::max(
            0.0F,
            std::min(
                y2,
                static_cast<float>(letterbox.original_height)
            )
        );

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        Detection detection;
        detection.class_id = best_class;
        detection.confidence = best_confidence;
        detection.x1 = x1;
        detection.y1 = y1;
        detection.x2 = x2;
        detection.y2 = y2;
        proposals.push_back(detection);
    }

    return class_aware_nms(
        proposals,
        class_count,
        iou_threshold,
        max_det
    );
}

double milliseconds_between(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

InferenceResult run_single_image(
    ncnn::Net& network,
    const std::string& image_path,
    int imgsz,
    int threads,
    int class_count,
    float confidence_threshold,
    float iou_threshold,
    int max_det
) {
    InferenceResult result;

    const auto io_start = std::chrono::steady_clock::now();
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    const auto io_end = std::chrono::steady_clock::now();

    result.timing.io_ms = milliseconds_between(io_start, io_end);

    if (image.empty()) {
        throw std::runtime_error("OpenCV failed to read image: " + image_path);
    }

    LetterboxInfo letterbox;

    const auto preprocess_start = std::chrono::steady_clock::now();
    ncnn::Mat input = preprocess_image(image, imgsz, letterbox);
    const auto preprocess_end = std::chrono::steady_clock::now();

    result.timing.preprocess_ms =
        milliseconds_between(preprocess_start, preprocess_end);

    ncnn::Extractor extractor = network.create_extractor();
    extractor.set_light_mode(true);

    const auto inference_start = std::chrono::steady_clock::now();

    const int input_status = extractor.input("in0", input);
    if (input_status != 0) {
        throw std::runtime_error(
            "NCNN failed to set input blob in0. Status=" +
            std::to_string(input_status)
        );
    }

    ncnn::Mat output;
    const int output_status = extractor.extract("out0", output);
    const auto inference_end = std::chrono::steady_clock::now();

    if (output_status != 0) {
        throw std::runtime_error(
            "NCNN failed to extract output blob out0. Status=" +
            std::to_string(output_status)
        );
    }

    result.timing.inference_ms =
        milliseconds_between(inference_start, inference_end);

    result.output_dims = output.dims;
    result.output_w = output.w;
    result.output_h = output.h;
    result.output_c = output.c;

    const auto postprocess_start = std::chrono::steady_clock::now();
    const OutputLayout layout =
        determine_output_layout(output, class_count);
    result.detections = decode_output(
        output,
        layout,
        letterbox,
        class_count,
        confidence_threshold,
        iou_threshold,
        max_det
    );
    const auto postprocess_end = std::chrono::steady_clock::now();

    result.timing.postprocess_ms =
        milliseconds_between(postprocess_start, postprocess_end);

    return result;
}

double timeval_seconds(const timeval& value) {
    return static_cast<double>(value.tv_sec) +
           static_cast<double>(value.tv_usec) / 1000000.0;
}

double process_cpu_seconds(const rusage& usage) {
    return timeval_seconds(usage.ru_utime) +
           timeval_seconds(usage.ru_stime);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::runtime_error("Cannot compute mean of an empty vector.");
    }

    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

double percentile(
    const std::vector<double>& values,
    double fraction
) {
    if (values.empty()) {
        throw std::runtime_error(
            "Cannot compute percentile of an empty vector."
        );
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    const double rank =
        fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower =
        static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper =
        static_cast<std::size_t>(std::ceil(rank));

    if (lower == upper) {
        return sorted[lower];
    }

    const double weight = rank - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) +
           sorted[upper] * weight;
}

void validate_model(
    ncnn::Net& network,
    const Options& options,
    const Metadata& metadata,
    const std::vector<std::string>& images
) {
    std::ofstream predictions(options.predictions_csv);
    if (!predictions) {
        throw std::runtime_error(
            "Unable to create predictions CSV: " +
            options.predictions_csv
        );
    }

    std::ofstream image_summary(options.images_csv);
    if (!image_summary) {
        throw std::runtime_error(
            "Unable to create images summary CSV: " +
            options.images_csv
        );
    }

    predictions
        << "image_path,image_name,image_width,image_height,"
        << "class_id,class_name,confidence,x1,y1,x2,y2\n";
    image_summary
        << "image_path,image_name,image_width,image_height,detections\n";

    bool output_shape_printed = false;
    std::size_t total_detections = 0;

    for (std::size_t index = 0; index < images.size(); ++index) {
        const InferenceResult result = run_single_image(
            network,
            images[index],
            options.imgsz,
            options.threads,
            static_cast<int>(metadata.class_names.size()),
            options.conf,
            options.iou,
            options.max_det
        );

        if (!output_shape_printed) {
            std::cout
                << "Output tensor: dims=" << result.output_dims
                << " w=" << result.output_w
                << " h=" << result.output_h
                << " c=" << result.output_c << '\n';
            output_shape_printed = true;
        }

        cv::Mat dimensions_image = cv::imread(
            images[index],
            cv::IMREAD_COLOR
        );
        if (dimensions_image.empty()) {
            throw std::runtime_error(
                "Failed to re-read image dimensions: " + images[index]
            );
        }

        const std::string filename = base_name(images[index]);

        image_summary
            << csv_escape(images[index]) << ','
            << csv_escape(filename) << ','
            << dimensions_image.cols << ','
            << dimensions_image.rows << ','
            << result.detections.size() << '\n';

        for (const Detection& detection : result.detections) {
            predictions
                << csv_escape(images[index]) << ','
                << csv_escape(filename) << ','
                << dimensions_image.cols << ','
                << dimensions_image.rows << ','
                << detection.class_id << ','
                << csv_escape(
                       metadata.class_names[
                           static_cast<std::size_t>(
                               detection.class_id
                           )
                       ]
                   ) << ','
                << std::fixed << std::setprecision(8)
                << detection.confidence << ','
                << detection.x1 << ','
                << detection.y1 << ','
                << detection.x2 << ','
                << detection.y2 << '\n';
        }

        total_detections += result.detections.size();

        std::cout
            << "\rValidate " << (index + 1)
            << "/" << images.size()
            << " detections=" << total_detections
            << std::flush;
    }

    std::cout << "\nValidation inference completed.\n"
              << "Predictions: " << options.predictions_csv << '\n'
              << "Image summary: " << options.images_csv << '\n';
}

void benchmark_model(
    ncnn::Net& network,
    const Options& options,
    const Metadata& metadata,
    const std::vector<std::string>& images
) {
    // Warm-up uses the same model and first benchmark image but is excluded
    // from latency and CPU calculations.
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
        run_single_image(
            network,
            images.front(),
            options.imgsz,
            options.threads,
            static_cast<int>(metadata.class_names.size()),
            options.conf,
            options.iou,
            options.max_det
        );
        std::cout
            << "\rWarm-up " << (iteration + 1)
            << "/" << options.warmup
            << std::flush;
    }
    if (options.warmup > 0) {
        std::cout << '\n';
    }

    std::vector<double> io_times;
    std::vector<double> preprocess_times;
    std::vector<double> inference_times;
    std::vector<double> postprocess_times;
    std::vector<double> compute_times;
    std::vector<double> total_times;

    const std::size_t measured_count =
        images.size() * static_cast<std::size_t>(options.repeat);

    io_times.reserve(measured_count);
    preprocess_times.reserve(measured_count);
    inference_times.reserve(measured_count);
    postprocess_times.reserve(measured_count);
    compute_times.reserve(measured_count);
    total_times.reserve(measured_count);

    rusage usage_before {};
    rusage usage_after {};

    if (getrusage(RUSAGE_SELF, &usage_before) != 0) {
        throw std::runtime_error(
            std::string("getrusage failed before benchmark: ") +
            std::strerror(errno)
        );
    }

    const double cpu_before = process_cpu_seconds(usage_before);
    const auto wall_start = std::chrono::steady_clock::now();

    bool output_shape_printed = false;
    InferenceResult last_result;
    std::size_t completed = 0;

    for (int repetition = 0; repetition < options.repeat; ++repetition) {
        for (const std::string& image_path : images) {
            InferenceResult result = run_single_image(
                network,
                image_path,
                options.imgsz,
                options.threads,
                static_cast<int>(metadata.class_names.size()),
                options.conf,
                options.iou,
                options.max_det
            );

            if (!output_shape_printed) {
                std::cout
                    << "Output tensor: dims=" << result.output_dims
                    << " w=" << result.output_w
                    << " h=" << result.output_h
                    << " c=" << result.output_c << '\n';
                output_shape_printed = true;
            }

            io_times.push_back(result.timing.io_ms);
            preprocess_times.push_back(result.timing.preprocess_ms);
            inference_times.push_back(result.timing.inference_ms);
            postprocess_times.push_back(result.timing.postprocess_ms);
            compute_times.push_back(result.timing.compute_ms());
            total_times.push_back(result.timing.total_ms());

            last_result = result;
            ++completed;

            std::cout
                << "\rBenchmark " << completed
                << "/" << measured_count
                << std::flush;
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();

    if (getrusage(RUSAGE_SELF, &usage_after) != 0) {
        throw std::runtime_error(
            std::string("getrusage failed after benchmark: ") +
            std::strerror(errno)
        );
    }

    std::cout << '\n';

    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();
    const double cpu_seconds =
        process_cpu_seconds(usage_after) - cpu_before;

    const long logical_cores_raw = sysconf(_SC_NPROCESSORS_ONLN);
    if (logical_cores_raw <= 0) {
        throw std::runtime_error(
            "Unable to determine logical CPU core count."
        );
    }
    const int logical_cores = static_cast<int>(logical_cores_raw);

    const double raw_cpu_percent =
        wall_seconds <= 0.0
            ? 0.0
            : cpu_seconds / wall_seconds * 100.0;
    const double normalized_cpu_percent =
        raw_cpu_percent / static_cast<double>(logical_cores);

    // Linux reports ru_maxrss in KiB.
    const double peak_rss_mib =
        static_cast<double>(usage_after.ru_maxrss) / 1024.0;

    const double mean_compute_ms = mean(compute_times);
    const double compute_fps =
        mean_compute_ms <= 0.0
            ? 0.0
            : 1000.0 / mean_compute_ms;
    const double total_fps =
        wall_seconds <= 0.0
            ? 0.0
            : static_cast<double>(measured_count) / wall_seconds;

    std::ofstream summary(options.summary_csv);
    if (!summary) {
        throw std::runtime_error(
            "Unable to create benchmark summary CSV: " +
            options.summary_csv
        );
    }

    summary
        << "model,classes,imgsz,threads,warmup,repeat,"
        << "benchmark_images,measured_inferences,conf,iou,max_det,"
        << "output_dims,output_w,output_h,output_c,"
        << "mean_io_ms,mean_preprocess_ms,mean_inference_ms,"
        << "mean_postprocess_ms,mean_compute_ms,"
        << "median_compute_ms,p95_compute_ms,compute_fps,"
        << "mean_total_ms,median_total_ms,p95_total_ms,total_fps,"
        << "peak_rss_mib,raw_process_cpu_percent,"
        << "normalized_board_cpu_percent,logical_cores\n";

    summary
        << csv_escape(base_name(options.model_dir)) << ','
        << metadata.class_names.size() << ','
        << options.imgsz << ','
        << options.threads << ','
        << options.warmup << ','
        << options.repeat << ','
        << images.size() << ','
        << measured_count << ','
        << std::fixed << std::setprecision(8)
        << options.conf << ','
        << options.iou << ','
        << options.max_det << ','
        << last_result.output_dims << ','
        << last_result.output_w << ','
        << last_result.output_h << ','
        << last_result.output_c << ','
        << mean(io_times) << ','
        << mean(preprocess_times) << ','
        << mean(inference_times) << ','
        << mean(postprocess_times) << ','
        << mean_compute_ms << ','
        << percentile(compute_times, 0.50) << ','
        << percentile(compute_times, 0.95) << ','
        << compute_fps << ','
        << mean(total_times) << ','
        << percentile(total_times, 0.50) << ','
        << percentile(total_times, 0.95) << ','
        << total_fps << ','
        << peak_rss_mib << ','
        << raw_cpu_percent << ','
        << normalized_cpu_percent << ','
        << logical_cores << '\n';

    std::cout
        << "Benchmark completed.\n"
        << "Summary: " << options.summary_csv << '\n'
        << "Median compute latency: "
        << percentile(compute_times, 0.50) << " ms\n"
        << "Compute FPS: " << compute_fps << '\n'
        << "Total FPS including image read: " << total_fps << '\n'
        << "Peak RSS: " << peak_rss_mib << " MiB\n"
        << "Normalized board CPU: "
        << normalized_cpu_percent << "%\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_options(argc, argv);

        const std::string param_path =
            join_path(options.model_dir, "model.ncnn.param");
        const std::string bin_path =
            join_path(options.model_dir, "model.ncnn.bin");
        const std::string metadata_path =
            join_path(options.model_dir, "metadata.yaml");

        if (!file_exists(param_path)) {
            throw std::runtime_error(
                "Missing NCNN parameter file: " + param_path
            );
        }
        if (!file_exists(bin_path)) {
            throw std::runtime_error(
                "Missing NCNN weight file: " + bin_path
            );
        }
        if (!file_exists(metadata_path)) {
            throw std::runtime_error(
                "Missing model metadata file: " + metadata_path
            );
        }

        const Metadata metadata = read_metadata(metadata_path);
        const std::vector<std::string> images =
            read_image_list(options.image_list, options.limit);

        ncnn::Net network;
        network.opt.use_vulkan_compute = false;
        network.opt.num_threads = options.threads;

        const int param_status =
            network.load_param(param_path.c_str());
        if (param_status != 0) {
            throw std::runtime_error(
                "NCNN load_param failed with status " +
                std::to_string(param_status)
            );
        }

        const int model_status =
            network.load_model(bin_path.c_str());
        if (model_status != 0) {
            throw std::runtime_error(
                "NCNN load_model failed with status " +
                std::to_string(model_status)
            );
        }

        std::cout
            << "Model: " << options.model_dir << '\n'
            << "Classes: " << metadata.class_names.size() << '\n'
            << "Images: " << images.size() << '\n'
            << "Mode: " << options.mode << '\n';

        if (options.mode == "validate") {
            validate_model(network, options, metadata, images);
        } else {
            benchmark_model(network, options, metadata, images);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        print_usage();
        return 1;
    }
}
