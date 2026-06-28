import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RESULTS_DIR = Path("./results")
RECORDINGS_DIR = Path("./recordings")
INTERPOLATED_RESULTS_DIR = RESULTS_DIR / "interpolated"
VISUALIZED_RESULTS_DIR = RECORDINGS_DIR / "visualized"

MODELS = [
    "YOLOv8n",
    "YOLO11n",
    "YOLO26n"
]

RECORDINGS = [
    "recording_1",
    "recording_2"
]


def draw_border(
        image,
        top_left,
        bottom_right,
        color=(0, 255, 0),
        thickness=10,
        line_length_x=200,
        line_length_y=200
):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(image, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(image, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(image, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(image, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(image, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(image, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(image, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return image


def parse_bbox(bbox_text):
    cleaned_text = (
        bbox_text
        .replace("[ ", "[")
        .replace("   ", " ")
        .replace("  ", " ")
        .replace(" ", ",")
    )

    return ast.literal_eval(cleaned_text)


def get_video_writer(video_capture, output_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def get_best_license_plates(results, video_capture):
    license_plates = {}

    for car_id in np.unique(results["car_id"]):
        car_results = results[results["car_id"] == car_id]
        max_score = np.amax(car_results["license_number_score"])

        best_row = car_results[
            car_results["license_number_score"] == max_score
            ].iloc[0]

        frame_number = int(best_row["frame_nmr"])
        license_plate_number = best_row["license_number"]

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video_capture.read()

        if not ret:
            continue

        x1, y1, x2, y2 = parse_bbox(best_row["license_plate_bbox"])

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        if license_crop.size == 0:
            continue

        license_crop = cv2.resize(
            license_crop,
            (int((x2 - x1) * 400 / (y2 - y1)), 400)
        )

        license_plates[car_id] = {
            "license_crop": license_crop,
            "license_plate_number": license_plate_number
        }

    return license_plates


def draw_license_plate_text(
        frame,
        license_plate_number,
        car_x1,
        car_x2,
        car_y1,
        crop_height
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4.3
    thickness = 17

    (text_width, text_height), _ = cv2.getTextSize(
        license_plate_number,
        font,
        font_scale,
        thickness
    )

    text_x = int((car_x2 + car_x1 - text_width) / 2)
    text_y = int(car_y1 - crop_height - 250 + (text_height / 2))

    cv2.putText(
        frame,
        license_plate_number,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )


def draw_license_plate_info(frame, row, license_plates):
    car_x1, car_y1, car_x2, car_y2 = parse_bbox(row["car_bbox"])
    x1, y1, x2, y2 = parse_bbox(row["license_plate_bbox"])

    draw_border(
        frame,
        (int(car_x1), int(car_y1)),
        (int(car_x2), int(car_y2)),
        color=(0, 255, 0),
        thickness=25,
        line_length_x=200,
        line_length_y=200
    )

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 0, 255),
        12
    )

    car_id = row["car_id"]

    if car_id not in license_plates:
        return

    license_crop = license_plates[car_id]["license_crop"]
    license_plate_number = license_plates[car_id]["license_plate_number"]

    crop_height, crop_width, _ = license_crop.shape

    crop_x1 = int((car_x2 + car_x1 - crop_width) / 2)
    crop_x2 = int((car_x2 + car_x1 + crop_width) / 2)

    crop_y1 = int(car_y1) - crop_height - 100
    crop_y2 = int(car_y1) - 100

    text_box_y1 = int(car_y1) - crop_height - 400
    text_box_y2 = int(car_y1) - crop_height - 100

    try:
        frame[crop_y1:crop_y2, crop_x1:crop_x2, :] = license_crop
        frame[text_box_y1:text_box_y2, crop_x1:crop_x2, :] = (255, 255, 255)

        draw_license_plate_text(
            frame,
            license_plate_number,
            car_x1,
            car_x2,
            car_y1,
            crop_height
        )

    except Exception:
        pass


def process_video(results, video_capture, video_writer, license_plates):
    frame_number = -1

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = video_capture.read()
        frame_number += 1

        if not ret:
            break

        frame_results = results[results["frame_nmr"] == frame_number]

        for row_index in range(len(frame_results)):
            row = frame_results.iloc[row_index]
            draw_license_plate_info(frame, row, license_plates)

        video_writer.write(frame)


def visualize_file(input_csv_path, input_video_path, output_video_path):
    if not input_csv_path.exists():
        print(f"Skipped missing CSV: {input_csv_path}")
        return

    if not input_video_path.exists():
        print(f"Skipped missing video: {input_video_path}")
        return

    results = pd.read_csv(input_csv_path)

    if len(results) == 0:
        print(f"Skipped empty CSV: {input_csv_path}")
        return

    video_capture = cv2.VideoCapture(str(input_video_path))
    video_writer = get_video_writer(video_capture, output_video_path)

    license_plates = get_best_license_plates(results, video_capture)

    process_video(
        results,
        video_capture,
        video_writer,
        license_plates
    )

    video_writer.release()
    video_capture.release()

    print(f"Created: {output_video_path}")


def main():
    VISUALIZED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        for recording_name in RECORDINGS:
            input_csv_path = INTERPOLATED_RESULTS_DIR / f"{model_name}_{recording_name}_interpolated.csv"
            input_video_path = RECORDINGS_DIR / f"{recording_name}.mp4"
            output_video_path = VISUALIZED_RESULTS_DIR / f"{model_name}_{recording_name}_visualized.mp4"

            visualize_file(
                input_csv_path,
                input_video_path,
                output_video_path
            )


if __name__ == "__main__":
    main()
