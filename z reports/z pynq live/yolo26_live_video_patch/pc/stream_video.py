"""PC-side synchronous video client for PYNQ-Z2 YOLO26 NCNN inference.

The PC reads a video frame, JPEG-encodes it, sends it to the PYNQ server,
waits for detections, draws boxes on the original frame, and displays it.
This first version is intentionally synchronous because it gives an exact
frame-to-detection match and is the fastest pipeline to validate correctly.
"""

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
        "--max-frames",
        type=int,
        default=0,
        help="0 means process the full video; use 3 first for a smoke test.",
    )
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in 1..65535")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in 1..100")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")

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

        frame_id = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

            encode_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
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

                left = int(round(x1))
                top = int(round(y1))
                right = int(round(x2))
                bottom = int(round(y2))
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

            cv2.imshow("PYNQ-Z2 YOLO26 NCNN", frame)
            print(
                "frame={} det={} preprocess={:.1f}ms inference={:.1f}ms "
                "postprocess={:.1f}ms server={:.1f}ms RTT={:.1f}ms".format(
                    frame_id,
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

            frame_id += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
