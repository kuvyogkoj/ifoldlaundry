
from transformers import Sam3Model, Sam3Processor
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import cv2

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam(device)
    prompt = "clothes"

    processed_frame = None
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to read webcam")
            break

        if processed_frame is None:
            cv2.imshow("Webcam", frame)
        else:
            cv2.imshow("Webcam", processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c"):
            mask_overlay = process_input(frame, model, processor, prompt, device)
            processed_frame = cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR)
    cap.release()
    cv2.destroyAllWindows()

            
def process_input(frame, model, processor, prompt, device) -> np.array:
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    inputs = processor(images=image_rgb, text=prompt, return_tensors="pt").to(device)

    with torch.inference_mode(): 
        output = model(**inputs)

    target_sizes = inputs["original_sizes"].cpu().tolist()
    segments = processor.post_process_instance_segmentation(outputs=output, threshold=0.5, mask_threshold=0.3, target_sizes=target_sizes)

    output_dir = Path("saved_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    return process_output(image_rgb, segments[0], output_dir, 0.5)

def process_output(frame, results, output_dir, alpha) -> np.array:
    overlay = np.asarray(frame).astype(np.float32).copy()
    for i in range(len(results["masks"])):
        mask = results["masks"][i]

        mask_bool = mask.cpu().numpy().astype(bool)

        color = np.array(np.random.randint(0, 256, 3), dtype=np.float32)
        overlay[mask_bool] = ( (1 - alpha) * overlay[mask_bool] + alpha * color)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    print("processed frame")
    return overlay

def load_sam(device) -> tuple[Sam3Processor, Sam3Model]:
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    return (processor, model)

if __name__ == "__main__":
    main()

