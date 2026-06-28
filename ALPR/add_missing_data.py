import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

RESULTS_DIR = Path("./results")
INTERPOLATED_RESULTS_DIR = RESULTS_DIR / "interpolated"

CSV_HEADER = [
    "frame_nmr",
    "car_id",
    "car_bbox",
    "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number",
    "license_number_score"
]

MODELS = [
    "YOLOv8n",
    "YOLO11n",
    "YOLO26n"
]

RECORDINGS = [
    "recording_1",
    "recording_2"
]


def parse_bbox(bbox_text):
    return list(map(float, bbox_text[1:-1].split()))


def format_bbox(bbox):
    return "[{} {} {} {}]".format(*bbox)


def load_csv(path):
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        return list(reader)


def write_csv(path, data):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(data)


def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row["frame_nmr"]) for row in data])
    car_ids = np.array([int(float(row["car_id"])) for row in data])

    car_bboxes = np.array([
        parse_bbox(row["car_bbox"])
        for row in data
    ])

    license_plate_bboxes = np.array([
        parse_bbox(row["license_plate_bbox"])
        for row in data
    ])

    interpolated_data = []

    for car_id in np.unique(car_ids):
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]

        original_rows = {
            int(row["frame_nmr"]): row
            for row in data
            if int(float(row["car_id"])) == int(float(car_id))
        }

        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]

        for index in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[index]
            car_bbox = car_bboxes[car_mask][index]
            license_plate_bbox = license_plate_bboxes[car_mask][index]

            if index > 0:
                previous_frame_number = car_frame_numbers[index - 1]
                previous_car_bbox = car_bboxes_interpolated[-1]
                previous_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - previous_frame_number > 1:
                    frames_gap = frame_number - previous_frame_number

                    x = np.array([previous_frame_number, frame_number])
                    x_new = np.linspace(
                        previous_frame_number,
                        frame_number,
                        num=frames_gap,
                        endpoint=False
                    )

                    car_interp_func = interp1d(
                        x,
                        np.vstack((previous_car_bbox, car_bbox)),
                        axis=0,
                        kind="linear"
                    )

                    license_plate_interp_func = interp1d(
                        x,
                        np.vstack((previous_license_plate_bbox, license_plate_bbox)),
                        axis=0,
                        kind="linear"
                    )

                    interpolated_car_bboxes = car_interp_func(x_new)
                    interpolated_license_plate_bboxes = license_plate_interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(
                        interpolated_license_plate_bboxes[1:]
                    )

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for index in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + index

            row = {
                "frame_nmr": str(frame_number),
                "car_id": str(car_id),
                "car_bbox": format_bbox(car_bboxes_interpolated[index]),
                "license_plate_bbox": format_bbox(
                    license_plate_bboxes_interpolated[index]
                )
            }

            if frame_number in original_rows:
                original_row = original_rows[frame_number]

                row["license_plate_bbox_score"] = original_row.get(
                    "license_plate_bbox_score",
                    "0"
                )
                row["license_number"] = original_row.get("license_number", "0")
                row["license_number_score"] = original_row.get(
                    "license_number_score",
                    "0"
                )
            else:
                row["license_plate_bbox_score"] = "0"
                row["license_number"] = "0"
                row["license_number_score"] = "0"

            interpolated_data.append(row)

    return interpolated_data


def process_file(input_path, output_path):
    if not input_path.exists():
        print(f"Skipped missing file: {input_path}")
        return

    data = load_csv(input_path)

    if len(data) == 0:
        print(f"Skipped empty file: {input_path}")
        return

    interpolated_data = interpolate_bounding_boxes(data)
    write_csv(output_path, interpolated_data)

    print(f"Created: {output_path}")


def main():
    INTERPOLATED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        for recording_name in RECORDINGS:
            input_path = RESULTS_DIR / f"{model_name}_{recording_name}.csv"
            output_path = INTERPOLATED_RESULTS_DIR / f"{model_name}_{recording_name}_interpolated.csv"
            process_file(input_path, output_path)


if __name__ == "__main__":
    main()
