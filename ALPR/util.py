import cv2
import string
import torch
import easyocr

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

dict_char_to_int = {
    "O": "0",
    "I": "1",
    "J": "3",
    "A": "4",
    "G": "6",
    "S": "5"
}

dict_int_to_char = {
    "0": "O",
    "1": "I",
    "3": "J",
    "4": "A",
    "6": "G",
    "5": "S"
}

romanian_patterns = [
    ["letter", "letter", "digit", "digit", "letter", "letter", "letter"],
    ["letter", "digit", "digit", "letter", "letter", "letter"],
    ["letter", "digit", "digit", "digit", "letter", "letter", "letter"]
]


def write_csv(results, output_path):
    with open(output_path, "w") as file:
        file.write(
            "frame_nmr,car_id,car_bbox,license_plate_bbox,"
            "license_plate_bbox_score,license_number,license_number_score\n"
        )

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                entry = results[frame_nmr][car_id]

                if (
                    "car" in entry
                    and "license_plate" in entry
                    and "text" in entry["license_plate"]
                ):
                    file.write(
                        "{},{},{},{},{},{},{}\n".format(
                            frame_nmr,
                            car_id,
                            "[{} {} {} {}]".format(*entry["car"]["bbox"]),
                            "[{} {} {} {}]".format(*entry["license_plate"]["bbox"]),
                            entry["license_plate"]["bbox_score"],
                            entry["license_plate"]["text"],
                            entry["license_plate"]["text_score"]
                        )
                    )


def char_matches_expected_type(char, expected_type):
    if expected_type == "letter":
        return char in string.ascii_uppercase or char in dict_int_to_char

    if expected_type == "digit":
        return char in string.digits or char in dict_char_to_int

    return False


def get_matching_pattern(text):
    for pattern in romanian_patterns:
        if len(text) != len(pattern):
            continue

        is_valid = True

        for char, expected_type in zip(text, pattern):
            if not char_matches_expected_type(char, expected_type):
                is_valid = False
                break

        if is_valid:
            return pattern

    return None


def license_complies_format(text):
    return get_matching_pattern(text) is not None


def format_license(text):
    pattern = get_matching_pattern(text)

    if pattern is None:
        return text

    formatted_text = ""

    for char, expected_type in zip(text, pattern):
        if expected_type == "letter" and char in dict_int_to_char:
            formatted_text += dict_int_to_char[char]

        elif expected_type == "digit" and char in dict_char_to_int:
            formatted_text += dict_char_to_int[char]

        else:
            formatted_text += char

    return formatted_text


def get_ocr_input_versions(license_plate_crop):
    crops = []

    crops.append(license_plate_crop)

    upscaled = cv2.resize(
        license_plate_crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )
    crops.append(upscaled)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    crops.append(gray)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    crops.append(thresh)

    return crops


def clean_license_text(text):
    return (
        text
        .upper()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
    )


def read_license_plate(license_plate_crop):
    crops = get_ocr_input_versions(license_plate_crop)

    best_text = None
    best_score = 0

    for crop in crops:
        detections = reader.readtext(
            crop,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False
        )

        for detection in detections:
            bbox, text, score = detection

            text = clean_license_text(text)

            if license_complies_format(text):
                formatted_text = format_license(text)

                if score > best_score:
                    best_text = formatted_text
                    best_score = score

    if best_text is not None:
        return best_text, best_score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle

    return -1, -1, -1, -1, -1
