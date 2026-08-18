"""PC-side synchronous video client for PYNQ-Z2 YOLO26 NCNN inference.

The PC reads a video frame, JPEG-encodes it, sends it to the PYNQ server,
waits for detections, draws boxes on the original frame, and displays it.
This first version is intentionally synchronous because it gives an exact
frame-to-detection match and is the fastest pipeline to validate correctly.
"""

'''
Pynq ssh:
bash 1_scripts/run_live_server.sh 4_layer_replacement_320_FP16

PC terminal:
& ".venv\Scripts\python.exe" `
  "stream_video_frame_skip.py" `
  --host 192.168.2.99 `
  --video "test.mov" `
  --send-size 320 `
  --display-width 1280 `
  --frame-skip 9

'''

import argparse
import socket
import struct
import time
from typing import List, Tuple

import cv2


CLASS_NAMES = [
    "articulated_truck",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "motorized_vehicle",
    "non_motorized_vehicle",
    "pedestrian",
    "pickup_truck",
    "single_unit_truck",
    "work_van",
]


def recv_line(sock: socket.socket) -> str:
    """Read one newline-terminated server record without losing TCP bytes."""
    data = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("PYNQ disconnected while returning detections.")
        if chunk == b"\n":
            return data.decode("utf-8")
        data.extend(chunk)


def receive_result(
    sock: socket.socket,
    expected_frame_id: int,
) -> Tuple[dict, List[Tuple[int, float, float, float, float, float]]]:
    """Receive the RESULT header followed by DET records and the END marker."""
    header = recv_line(sock).split()
    if len(header) != 7 or header[0] != "RESULT":
        raise RuntimeError("Unexpected server response: " + " ".join(header))

    frame_id = int(header[1])
    if frame_id != expected_frame_id:
        raise RuntimeError(
            "Frame mismatch: sent {}, received {}".format(expected_frame_id, frame_id)
        )

    timing = {
        "server_total_ms": float(header[2]),
        "preprocess_ms": float(header[3]),
        "inference_ms": float(header[4]),
        "postprocess_ms": float(header[5]),
        "count": int(header[6]),
    }

    detections = []
    while True:
        line = recv_line(sock)
        if line == "END":
            break

        fields = line.split()
        if len(fields) != 7 or fields[0] != "DET":
            raise RuntimeError("Unexpected detection record: " + line)

        detections.append(
            (
                int(fields[1]),
                float(fields[2]),
                float(fields[3]),
                float(fields[4]),
                float(fields[5]),
                float(fields[6]),
            )
        )

    if len(detections) != timing["count"]:
        raise RuntimeError(
            "Server reported {} detections but sent {}.".format(
                timing["count"], len(detections)
            )
        )

    return timing, detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send PC video frames to PYNQ-Z2 and display returned YOLO boxes."
    )
    parser.add_argument("--host", required=True, help="PYNQ IP address, e.g. 192.168.2.99")
    parser.add_argument("--video", required=True, help="Path to the video file on this PC")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--send-size",
        type=int,
        required=True,
        help=(
            "Maximum width/height sent to PYNQ. Use 320 for "
            "4_layer_replacement_320_FP16 and 640 for 2_baseline_full."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 means process the full video; use 3 first for a smoke test.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help=(
            "Width of the PC preview window in pixels. The display keeps the "
            "video aspect ratio. Example: 1280 for a smaller 16:9 preview."
        ),
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help=(
            "Number of source-video frames to skip after each processed frame. "
            "0 = process every frame, 1 = process every 2nd frame, "
            "2 = process every 3rd frame, etc."
        ),
    )
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in 1..65535")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in 1..100")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if args.send_size <= 0:
        raise ValueError("--send-size must be positive")
    if args.display_width <= 0:
        raise ValueError("--display-width must be positive")
    if args.frame_skip < 0:
        raise ValueError("--frame-skip must be 0 or greater")

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open video: {}".format(args.video))

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video: {}".format(args.video))
    print("Source FPS: {:.3f}".format(source_fps))
    print("Source frames: {}".format(source_frames))
    print("Connecting to PYNQ {}:{} ...".format(args.host, args.port))

    with socket.create_connection((args.host, args.port), timeout=15.0) as sock:
        # Inference can take ~10 s on the baseline model, so do not use a short
        # per-frame timeout after the TCP connection has been established.
        sock.settimeout(None)
        print("Connected. Press Q in the video window to stop.")
        print(
            "Frame skip: {} (processing 1 of every {} source frames)".format(
                args.frame_skip,
                args.frame_skip + 1,
            )
        )

        processed_count = 0
        source_frame_id = 0

        while True:
            if args.max_frames > 0 and processed_count >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            # Keep the original source-video frame number in the protocol/logs so
            # it remains clear which frame was actually processed after skipping.
            frame_id = source_frame_id

            # Downscale on the PC before JPEG encoding. This preserves aspect ratio
            # and avoids making the PYNQ decode and resize a full 4K frame.
            original_height, original_width = frame.shape[:2]
            resize_scale = args.send_size / float(max(original_width, original_height))
            send_width = int(round(original_width * resize_scale))
            send_height = int(round(original_height * resize_scale))
            send_frame = cv2.resize(
                frame,
                (send_width, send_height),
                interpolation=cv2.INTER_AREA,
            )

            encode_ok, encoded = cv2.imencode(
                ".jpg",
                send_frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not encode_ok:
                raise RuntimeError("OpenCV failed to JPEG-encode frame {}.".format(frame_id))

            payload = encoded.tobytes()
            # The two uint32 values are network-endian to match ntohl() on PYNQ.
            request_header = struct.pack("!II", frame_id, len(payload))

            round_trip_start = time.perf_counter()
            sock.sendall(request_header)
            sock.sendall(payload)
            timing, detections = receive_result(sock, frame_id)
            round_trip_ms = (time.perf_counter() - round_trip_start) * 1000.0

            for class_id, confidence, x1, y1, x2, y2 in detections:
                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    raise RuntimeError("Unknown class id returned by PYNQ: {}".format(class_id))

                # PYNQ returns coordinates relative to the smaller frame it received.
                # Scale them back to the original PC video frame before drawing.
                x_scale = original_width / float(send_width)
                y_scale = original_height / float(send_height)
                left = int(round(x1 * x_scale))
                top = int(round(y1 * y_scale))
                right = int(round(x2 * x_scale))
                bottom = int(round(y2 * y_scale))
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                label = "{} {:.2f}".format(CLASS_NAMES[class_id], confidence)
                text_y = max(20, top - 6)
                cv2.putText(
                    frame,
                    label,
                    (left, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            status = (
                "frame={} det={} inference={:.0f}ms RTT={:.0f}ms"
                .format(frame_id, len(detections), timing["inference_ms"], round_trip_ms)
            )
            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # The original video can be 4K, which is larger than many PC screens.
            # Resize only the DISPLAY copy so the entire frame is visible while the
            # original frame still keeps full-resolution bounding-box coordinates.
            display_scale = args.display_width / float(original_width)
            display_height = int(round(original_height * display_scale))
            display_frame = cv2.resize(
                frame,
                (args.display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("PYNQ-Z2 YOLO26 NCNN", display_frame)
            print(
                "frame={} sent={}x{} det={} preprocess={:.1f}ms inference={:.1f}ms "
                "postprocess={:.1f}ms server={:.1f}ms RTT={:.1f}ms".format(
                    frame_id,
                    send_width,
                    send_height,
                    len(detections),
                    timing["preprocess_ms"],
                    timing["inference_ms"],
                    timing["postprocess_ms"],
                    timing["server_total_ms"],
                    round_trip_ms,
                )
            )

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

            processed_count += 1

            # Advance over skipped frames without decoding them. This avoids
            # unnecessary PC processing, network transfer, and PYNQ inference.
            skipped = 0
            for _ in range(args.frame_skip):
                if not capture.grab():
                    break
                skipped += 1

            source_frame_id += 1 + skipped

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
