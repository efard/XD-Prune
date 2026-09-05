#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <net.h>

namespace {

// This server intentionally keeps the same preprocessing, YOLO26 output decode,
// and class-aware NMS logic as the validated image runner. The networking layer
// only changes where an image comes from and where detections are returned.
struct Options {
    std::string model_dir;
    int imgsz = 640;
    int threads = 2;
    int port = 5000;
    int max_det = 300;
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
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
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
    if (!left.empty() && left.back() == '/') {
        return left + right;
    }
    return left + "/" + right;
}

Metadata read_metadata(const std::string& metadata_path) {
    std::ifstream input(metadata_path);
    if (!input) {
        throw std::runtime_error("Unable to open metadata file: " + metadata_path);
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
        std::string value_text = trim(stripped.substr(colon + 1));
        if (value_text.size() >= 2 &&
            ((value_text.front() == '\'' && value_text.back() == '\'') ||
             (value_text.front() == '"' && value_text.back() == '"'))) {
            value_text = value_text.substr(1, value_text.size() - 2);
        }

        std::size_t consumed = 0;
        const int class_id = std::stoi(key_text, &consumed);
        if (consumed != key_text.size() || class_id < 0) {
            throw std::runtime_error("Invalid class entry in metadata: " + stripped);
        }
        indexed_names[class_id] = value_text;
    }

    if (indexed_names.empty()) {
        throw std::runtime_error("No class names found in metadata: " + metadata_path);
    }

    const int highest_id = indexed_names.rbegin()->first;
    metadata.class_names.resize(static_cast<std::size_t>(highest_id + 1));
    for (int class_id = 0; class_id <= highest_id; ++class_id) {
        const auto found = indexed_names.find(class_id);
        if (found == indexed_names.end()) {
            throw std::runtime_error("Metadata class ids must be continuous from zero.");
        }
        metadata.class_names[static_cast<std::size_t>(class_id)] = found->second;
    }

    return metadata;
}

ncnn::Mat preprocess_image(const cv::Mat& original, int imgsz, LetterboxInfo& info) {
    if (original.empty()) {
        throw std::runtime_error("Cannot preprocess an empty image.");
    }

    info.original_width = original.cols;
    info.original_height = original.rows;
    info.scale = std::min(
        static_cast<float>(imgsz) / static_cast<float>(info.original_width),
        static_cast<float>(imgsz) / static_cast<float>(info.original_height)
    );

    const int resized_width = static_cast<int>(
        std::round(static_cast<float>(info.original_width) * info.scale)
    );
    const int resized_height = static_cast<int>(
        std::round(static_cast<float>(info.original_height) * info.scale)
    );

    const int horizontal_padding = imgsz - resized_width;
    const int vertical_padding = imgsz - resized_height;

    // Match Ultralytics centered letterbox rounding used by the existing runner.
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
        throw std::runtime_error("Letterbox preprocessing produced an unexpected size.");
    }
    if (!letterboxed.isContinuous()) {
        letterboxed = letterboxed.clone();
    }

    // OpenCV is BGR, while YOLO expects RGB. Normalize to [0, 1].
    ncnn::Mat input = ncnn::Mat::from_pixels(
        letterboxed.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        imgsz,
        imgsz
    );
    const float normalization[3] = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F
    };
    input.substract_mean_normalize(nullptr, normalization);
    return input;
}

OutputLayout determine_output_layout(const ncnn::Mat& output, int class_count) {
    const int expected_features = class_count + 4;
    if (output.elempack != 1 || output.dims != 2) {
        std::ostringstream message;
        message << "Unsupported output tensor: dims=" << output.dims
                << " w=" << output.w
                << " h=" << output.h
                << " c=" << output.c
                << " elempack=" << output.elempack;
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
    message << "Output tensor does not match metadata class count. Classes="
            << class_count << ", expected features=" << expected_features
            << ", output w=" << output.w << ", output h=" << output.h;
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

float intersection_over_union(const Detection& left, const Detection& right) {
    const float intersection_left = std::max(left.x1, right.x1);
    const float intersection_top = std::max(left.y1, right.y1);
    const float intersection_right = std::min(left.x2, right.x2);
    const float intersection_bottom = std::min(left.y2, right.y2);

    const float intersection_width =
        std::max(0.0F, intersection_right - intersection_left);
    const float intersection_height =
        std::max(0.0F, intersection_bottom - intersection_top);
    const float intersection_area = intersection_width * intersection_height;

    const float left_area =
        std::max(0.0F, left.x2 - left.x1) *
        std::max(0.0F, left.y2 - left.y1);
    const float right_area =
        std::max(0.0F, right.x2 - right.x1) *
        std::max(0.0F, right.y2 - right.y1);
    const float union_area = left_area + right_area - intersection_area;

    return union_area <= 0.0F ? 0.0F : intersection_area / union_area;
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
        grouped[static_cast<std::size_t>(detection.class_id)].push_back(detection);
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

        std::vector<unsigned char> suppressed(class_detections.size(), 0);
        for (std::size_t index = 0; index < class_detections.size(); ++index) {
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

        // end2end=false export already decodes DFL/anchors/stride and sigmoid.
        const float center_x = output_value(output, layout, 0, anchor);
        const float center_y = output_value(output, layout, 1, anchor);
        const float width = output_value(output, layout, 2, anchor);
        const float height = output_value(output, layout, 3, anchor);

        float x1 = center_x - width / 2.0F;
        float y1 = center_y - height / 2.0F;
        float x2 = center_x + width / 2.0F;
        float y2 = center_y + height / 2.0F;

        // Convert boxes from model input coordinates back to the received frame.
        x1 = (x1 - static_cast<float>(letterbox.pad_left)) / letterbox.scale;
        y1 = (y1 - static_cast<float>(letterbox.pad_top)) / letterbox.scale;
        x2 = (x2 - static_cast<float>(letterbox.pad_left)) / letterbox.scale;
        y2 = (y2 - static_cast<float>(letterbox.pad_top)) / letterbox.scale;

        x1 = std::max(0.0F, std::min(x1, static_cast<float>(letterbox.original_width)));
        y1 = std::max(0.0F, std::min(y1, static_cast<float>(letterbox.original_height)));
        x2 = std::max(0.0F, std::min(x2, static_cast<float>(letterbox.original_width)));
        y2 = std::max(0.0F, std::min(y2, static_cast<float>(letterbox.original_height)));

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

    return class_aware_nms(proposals, class_count, iou_threshold, max_det);
}

InferenceResult run_inference(
    ncnn::Net& network,
    const cv::Mat& image,
    int imgsz,
    int class_count,
    float confidence_threshold,
    float iou_threshold,
    int max_det
) {
    InferenceResult result;
    LetterboxInfo letterbox;

    const auto preprocess_start = std::chrono::steady_clock::now();
    ncnn::Mat input = preprocess_image(image, imgsz, letterbox);
    const auto preprocess_end = std::chrono::steady_clock::now();

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

    const auto postprocess_start = std::chrono::steady_clock::now();
    const OutputLayout layout = determine_output_layout(output, class_count);
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

    result.preprocess_ms = std::chrono::duration<double, std::milli>(
        preprocess_end - preprocess_start
    ).count();
    result.inference_ms = std::chrono::duration<double, std::milli>(
        inference_end - inference_start
    ).count();
    result.postprocess_ms = std::chrono::duration<double, std::milli>(
        postprocess_end - postprocess_start
    ).count();
    return result;
}

// TCP is a byte stream, so one recv/send call is not guaranteed to transfer
// the requested amount. These helpers continue until the exact payload is done.
bool receive_exact(int socket_fd, void* buffer, std::size_t size, bool allow_clean_eof) {
    unsigned char* cursor = static_cast<unsigned char*>(buffer);
    std::size_t received = 0;
    while (received < size) {
        const ssize_t count = recv(socket_fd, cursor + received, size - received, 0);
        if (count == 0) {
            if (allow_clean_eof && received == 0) {
                return false;
            }
            throw std::runtime_error("Client disconnected during a frame transfer.");
        }
        if (count < 0) {
            throw std::runtime_error("recv() failed.");
        }
        received += static_cast<std::size_t>(count);
    }
    return true;
}

void send_exact(int socket_fd, const std::string& text) {
    const unsigned char* cursor = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t sent = 0;
    while (sent < text.size()) {
        const ssize_t count = send(socket_fd, cursor + sent, text.size() - sent, 0);
        if (count <= 0) {
            throw std::runtime_error("send() failed.");
        }
        sent += static_cast<std::size_t>(count);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Options options;

        // Argument parsing is kept explicit so the selected model size is never
        // guessed. Use --imgsz 640 for baseline and --imgsz 320 for the optimized model.
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help" || argument == "-h") {
                std::cout
                    << "Usage: yolo26_ncnn_server --model-dir DIR --imgsz N "
                    << "[--threads 2] [--port 5000] [--conf 0.25] "
                    << "[--iou 0.70] [--max-det 300]\n";
                return 0;
            }
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value after argument: " + argument);
            }
            const std::string value = argv[++index];

            if (argument == "--model-dir") {
                options.model_dir = value;
            } else if (argument == "--imgsz") {
                options.imgsz = std::stoi(value);
            } else if (argument == "--threads") {
                options.threads = std::stoi(value);
            } else if (argument == "--port") {
                options.port = std::stoi(value);
            } else if (argument == "--conf") {
                options.conf = std::stof(value);
            } else if (argument == "--iou") {
                options.iou = std::stof(value);
            } else if (argument == "--max-det") {
                options.max_det = std::stoi(value);
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }

        if (options.model_dir.empty()) {
            throw std::runtime_error("--model-dir is required.");
        }
        if (options.imgsz <= 0 || options.threads <= 0 || options.max_det <= 0) {
            throw std::runtime_error("imgsz, threads, and max-det must be positive.");
        }
        if (options.port <= 0 || options.port > 65535) {
            throw std::runtime_error("--port must be in 1..65535.");
        }
        if (options.conf < 0.0F || options.conf > 1.0F) {
            throw std::runtime_error("--conf must be in [0, 1].");
        }
        if (options.iou <= 0.0F || options.iou > 1.0F) {
            throw std::runtime_error("--iou must be in (0, 1].");
        }

        const std::string param_path = join_path(options.model_dir, "model.ncnn.param");
        const std::string bin_path = join_path(options.model_dir, "model.ncnn.bin");
        const std::string metadata_path = join_path(options.model_dir, "metadata.yaml");
        if (!file_exists(param_path) || !file_exists(bin_path) || !file_exists(metadata_path)) {
            throw std::runtime_error("model.ncnn.param/bin/metadata.yaml must all exist in --model-dir.");
        }

        const Metadata metadata = read_metadata(metadata_path);

        // Load the model once. Every subsequent frame reuses this same NCNN Net.
        ncnn::Net network;
        network.opt.use_vulkan_compute = false;
        network.opt.num_threads = options.threads;
        const int param_status = network.load_param(param_path.c_str());
        if (param_status != 0) {
            throw std::runtime_error("NCNN load_param failed: " + std::to_string(param_status));
        }
        const int model_status = network.load_model(bin_path.c_str());
        if (model_status != 0) {
            throw std::runtime_error("NCNN load_model failed: " + std::to_string(model_status));
        }

        const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            throw std::runtime_error("socket() failed.");
        }
        int reuse_address = 1;
        if (setsockopt(
                server_fd,
                SOL_SOCKET,
                SO_REUSEADDR,
                &reuse_address,
                sizeof(reuse_address)
            ) != 0) {
            close(server_fd);
            throw std::runtime_error("setsockopt(SO_REUSEADDR) failed.");
        }

        sockaddr_in address {};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(static_cast<uint16_t>(options.port));

        if (bind(
                server_fd,
                reinterpret_cast<sockaddr*>(&address),
                sizeof(address)
            ) != 0) {
            close(server_fd);
            throw std::runtime_error("bind() failed. Check whether the port is already in use.");
        }
        if (listen(server_fd, 1) != 0) {
            close(server_fd);
            throw std::runtime_error("listen() failed.");
        }

        std::cout
            << "YOLO26 NCNN live server ready\n"
            << "  model: " << options.model_dir << '\n'
            << "  imgsz: " << options.imgsz << '\n'
            << "  classes: " << metadata.class_names.size() << '\n'
            << "  threads: " << options.threads << '\n'
            << "  port: " << options.port << '\n'
            << "Waiting for PC connection..." << std::endl;

        sockaddr_in client_address {};
        socklen_t client_length = sizeof(client_address);
        const int client_fd = accept(
            server_fd,
            reinterpret_cast<sockaddr*>(&client_address),
            &client_length
        );
        if (client_fd < 0) {
            close(server_fd);
            throw std::runtime_error("accept() failed.");
        }

        char client_ip[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &client_address.sin_addr, client_ip, sizeof(client_ip));
        std::cout << "PC connected from " << client_ip << std::endl;

        // Request protocol: [uint32 frame_id][uint32 jpeg_size][JPEG bytes],
        // where both uint32 values use network byte order.
        const std::size_t max_jpeg_bytes = 20U * 1024U * 1024U;
        while (true) {
            uint32_t header[2] = {0, 0};
            if (!receive_exact(client_fd, header, sizeof(header), true)) {
                std::cout << "PC disconnected normally." << std::endl;
                break;
            }

            const uint32_t frame_id = ntohl(header[0]);
            const uint32_t jpeg_size = ntohl(header[1]);
            if (jpeg_size == 0 || static_cast<std::size_t>(jpeg_size) > max_jpeg_bytes) {
                throw std::runtime_error("Invalid JPEG payload size received.");
            }

            std::vector<unsigned char> jpeg(static_cast<std::size_t>(jpeg_size));
            receive_exact(client_fd, jpeg.data(), jpeg.size(), false);

            const auto frame_start = std::chrono::steady_clock::now();
            cv::Mat image = cv::imdecode(jpeg, cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("OpenCV failed to decode the received JPEG frame.");
            }

            const InferenceResult result = run_inference(
                network,
                image,
                options.imgsz,
                static_cast<int>(metadata.class_names.size()),
                options.conf,
                options.iou,
                options.max_det
            );
            const auto frame_end = std::chrono::steady_clock::now();
            const double server_total_ms = std::chrono::duration<double, std::milli>(
                frame_end - frame_start
            ).count();

            // Response is line-oriented text so the PC can inspect/debug it easily.
            // RESULT frame_id server_total_ms preprocess_ms inference_ms postprocess_ms count
            std::ostringstream response;
            response << std::fixed << std::setprecision(6)
                     << "RESULT " << frame_id << ' '
                     << server_total_ms << ' '
                     << result.preprocess_ms << ' '
                     << result.inference_ms << ' '
                     << result.postprocess_ms << ' '
                     << result.detections.size() << '\n';

            for (const Detection& detection : result.detections) {
                response << "DET "
                         << detection.class_id << ' '
                         << detection.confidence << ' '
                         << detection.x1 << ' '
                         << detection.y1 << ' '
                         << detection.x2 << ' '
                         << detection.y2 << '\n';
            }
            response << "END\n";
            send_exact(client_fd, response.str());

            std::cout
                << "frame=" << frame_id
                << " detections=" << result.detections.size()
                << " inference_ms=" << std::fixed << std::setprecision(2)
                << result.inference_ms
                << " server_total_ms=" << server_total_ms
                << std::endl;
        }

        close(client_fd);
        close(server_fd);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
