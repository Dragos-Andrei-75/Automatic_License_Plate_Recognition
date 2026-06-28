import cv2
from pathlib import Path
from ultralytics import YOLO
from util import get_car, read_license_plate, write_csv

# Pipeline:
# 1.  Load Model
# 2.  Load Video
# 3.  Read Frames
# 4.  Detect & Track Vehicles
# 5.  Detect License Plates
# 6.  Assign License Plate to Car
# 7.  Crop License Plate
# 8.  Process License Plate
# 9.  Read License Plate Number
# 10. Write Results

coco_model = YOLO("yolov8n.pt")


def process_video(model_name, model_path, video_name, video_path, output_path):
    print("\n" + "=" * 12)
    print(f"Model: {model_name}")
    print(f"Video: {video_name}")
    print("=" * 12)

    results = {}

    # 1. Load Model
    license_plate_detector = YOLO(model_path)

    # 2. Load Video
    cap = cv2.VideoCapture(video_path)

    # COCO Classes for Vehicles (Car, Bus, Truck)
    vehicles = [2, 5, 7]

    # 3. Read Frames
    ret = True
    frame_nmr = 0

    while ret:
        ret, frame = cap.read()

        if ret:
            results[frame_nmr] = {}

            # 4. Detect & Track Vehicles
            track_results = coco_model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=vehicles,
                verbose=False
            )[0]

            track_ids = []

            if track_results.boxes.id is not None:
                boxes = track_results.boxes.xyxy.cpu().numpy()
                ids = track_results.boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, ids):
                    track_ids.append([box[0], box[1], box[2], box[3], track_id])

            # 5. Detect License Plates
            license_plates = license_plate_detector.predict(
                frame,
                conf=0.10,
                device=0,
                verbose=False
            )[0]

            print(f"{model_name} | {video_name} | Frame {frame_nmr}: plates detected = {len(license_plates.boxes)}")

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # 6. Assign License Plate to Car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # 7. Crop License Plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    # 9. Read License Plate Number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            "car": {
                                "bbox": [xcar1, ycar1, xcar2, ycar2]
                            },
                            "license_plate": {
                                "bbox": [x1, y1, x2, y2],
                                "text": license_plate_text,
                                "bbox_score": score,
                                "text_score": license_plate_text_score
                            }
                        }

        frame_nmr += 1

    write_csv(results, output_path)
    cap.release()


def main():
    Path("./results").mkdir(exist_ok=True)

    models = {
        "YOLOv8n": "./best_models/YOLOv8n_best.pt",
        "YOLO11n": "./best_models/YOLO11n_best.pt",
        "YOLO26n": "./best_models/YOLO26n_best.pt"
    }

    videos = {
        "recording_1": "./recordings/recording_1.mp4",
        "recording_2": "./recordings/recording_2.mp4"
    }

    for model_name, model_path in models.items():
        for video_name, video_path in videos.items():
            output_path = f"./results/{model_name}_{video_name}.csv"

            process_video(
                model_name=model_name,
                model_path=model_path,
                video_name=video_name,
                video_path=video_path,
                output_path=output_path
            )


if __name__ == "__main__":
    main()
