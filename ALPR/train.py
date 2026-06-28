import gc
import torch
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO


def train():
    # Base Models
    models = {
        "YOLOv8n": "yolov8n.pt",
        "YOLO11n": "yolo11n.pt",
        "YOLO26n": "yolo26n.pt"
    }

    # Path to Dataset
    dataset_path = "./dataset/data.yaml"

    # Results Table
    results = []

    # Train, Validate and Test Each Model
    for model_name in models:

        print("\n")
        print("=" * 7)
        print(f"{model_name}")
        print("=" * 7)

        # Load Model
        model = YOLO(models[model_name])

        # Train Model
        model.train(
            data=dataset_path,
            epochs=10,
            imgsz=640,
            device=0,
            batch=4,
            workers=0,
            augment=True,
            verbose=True,
            save=True,
            project="training",
            name=model_name,
            exist_ok=True
        )

        # Keep Best Model
        best_model = YOLO(f"./runs/detect/training/{model_name}/weights/best.pt")

        # Validate Best Model
        test_metrics = best_model.val(
            data=dataset_path,
            split="test",
            save=True,
            project="validation",
            name=model_name,
            exist_ok=True
        )

        # Extract Test Metrics
        precision = test_metrics.box.mp
        recall = test_metrics.box.mr
        map50 = test_metrics.box.map50
        map5095 = test_metrics.box.map

        # Calculate F1 Score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        # Display Test Metrics
        print(f"\nResults for {model_name}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"mAP50     : {map50:.4f}")
        print(f"mAP50-95  : {map5095:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        # Test Best Model
        test_directory = Path(dataset_path).parent / "test" / "images"

        best_model.predict(
            source=str(test_directory),
            conf=0.25,
            save=True,
            project="predictions",
            name=model_name,
            exist_ok=True
        )

        # Save Best Model
        best_model.save(f"./best_models/{model_name}_best.pt")

        # Store Results
        results.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "mAP50": map50,
            "mAP50-95": map5095,
            "F1": f1
        })

        # Cleanup
        del model
        del best_model
        torch.cuda.empty_cache()
        gc.collect()

    # Final Comparison Table
    print("\n")
    print("=" * 16)
    print("FINAL COMPARISON")
    print("=" * 16)

    for result in results:
        print(
            f"{result['Model']:8} | "
            f"P={result['Precision']:.4f} | "
            f"R={result['Recall']:.4f} | "
            f"mAP50={result['mAP50']:.4f} | "
            f"mAP50-95={result['mAP50-95']:.4f} | "
            f"F1={result['F1']:.4f}"
        )


if __name__ == "__main__":
    freeze_support()
    train()
