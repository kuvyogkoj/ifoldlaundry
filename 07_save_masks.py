import numpy as np
import torch
from PIL import Image
from pathlib import Path
from transformers import Sam3Processor, Sam3Model

def main() -> None:
    # Open image
    image = Image.open("input.jpg").convert("RGB")
    # Load SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam(device)
    model.eval()
    # Create inputs/ouputs
    inputs = processor(images=image, text="buildings", return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    target_size = [image.size[::-1]]
    segments = processor.post_process_instance_segmentation(outputs=outputs, threshold=0.5, mask_threshold=0.3, target_sizes=target_size)


    # create directory and save masks
    output_dir = Path("saved_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(segments)):
        save_masks(segments[i], output_dir)
    
    
def save_masks(results, output_dir) -> None:
    
    for i in range(len(results["masks"])):
        file_path = output_dir / f"instance{i:04d}.png"
        mask = results["masks"][i]
        mask_array = mask.cpu().numpy()
        mask_array = mask_array.astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array)
        mask_image.save(file_path, "PNG")
        print(f"Saved instance{i} at {file_path}")

def load_sam(device) -> tuple[Sam3Processor, Sam3Model]:
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    return (processor, model)


if __name__ == "__main__":
    main()