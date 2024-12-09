from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("yolo11m.pt")

# Image path
path = "1EYFejGUjvjPcc4PZTwoufw.jpg"

# YOLO inference
results = model(source=path, conf=0.4)

# image with YOLO annotations
result_img = results[0].plot()[..., ::-1] # Convert from BGR to RGB

plt.imshow(result_img)
plt.axis('off')
plt.show()