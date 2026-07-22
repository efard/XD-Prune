#include <filesystem>
#include <iostream>
#include <string>

#include <net.h>

int main(int argc, char* argv[]) {
    // Require the exported model directory explicitly so the program never
    // loads a model from an unintended working directory.
    if (argc != 2) {
        std::cerr << "Usage: yolo26_ncnn <ncnn_model_directory>\n";
        return 1;
    }

    const std::filesystem::path model_directory = argv[1];
    const std::filesystem::path param_path =
        model_directory / "model.ncnn.param";
    const std::filesystem::path bin_path =
        model_directory / "model.ncnn.bin";
    const std::filesystem::path metadata_path =
        model_directory / "metadata.yaml";

    // Check every required deployment artifact before calling NCNN. This
    // produces a clear error instead of an ambiguous model-loading failure.
    if (!std::filesystem::is_regular_file(param_path)) {
        std::cerr << "Missing NCNN parameter file: " << param_path << '\n';
        return 2;
    }

    if (!std::filesystem::is_regular_file(bin_path)) {
        std::cerr << "Missing NCNN weight file: " << bin_path << '\n';
        return 3;
    }

    if (!std::filesystem::is_regular_file(metadata_path)) {
        std::cerr << "Missing Ultralytics metadata file: "
                  << metadata_path << '\n';
        return 4;
    }

    ncnn::Net network;

    // The PYNQ ARM-side baseline uses NCNN's CPU backend. FPGA acceleration,
    // if added later, is a separate hardware/software co-design stage.
    network.opt.use_vulkan_compute = false;

    // load_param() reconstructs the network graph and load_model() loads the
    // trained tensors. A return value other than zero means NCNN rejected the
    // corresponding file.
    const int param_status =
        network.load_param(param_path.string().c_str());
    if (param_status != 0) {
        std::cerr << "Failed to load model.ncnn.param. NCNN status: "
                  << param_status << '\n';
        return 5;
    }

    const int model_status =
        network.load_model(bin_path.string().c_str());
    if (model_status != 0) {
        std::cerr << "Failed to load model.ncnn.bin. NCNN status: "
                  << model_status << '\n';
        return 6;
    }

    std::cout << "NCNN model load: PASS\n"
              << "Parameter file: " << param_path << '\n'
              << "Weight file:    " << bin_path << '\n'
              << "Metadata file:  " << metadata_path << '\n';

    // This first-stage executable deliberately stops after loading the model.
    // Image preprocessing, output decoding, confidence filtering, and NMS
    // must be implemented against the actual exported YOLO26 output blobs.
    return 0;
}
