from ultralytics import YOLO
import torch

print(torch.cuda.is_available())

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

if __name__ == '__main__':
# Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="./datasets/Pistols/data.yaml", epochs=10, device="0", workers=0, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image

    model.export(format="onnx", dynamic=True)