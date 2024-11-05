from ultralytics import YOLO

# model = YOLO('models/generic/best.pt')
model = YOLO('yolov8x')

results = model.predict('input_videos/output_video_000.mp4', save=True)
print(results[0])
print('========================================')
for box in results[0].boxes:
    print(box)