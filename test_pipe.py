import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

# --- 1. Configuration ---
YOLO_MODEL_PATH = r'C:\Users\david\Desktop\Thesis Test\besttrained.pt'
EFFNET_MODEL_PATH = r'C:\Users\david\Desktop\Thesis Test\best_pineapple_classifier.pth'
TEST_IMAGE_PATH = r'C:\Users\david\Desktop\Thesis Test\pine2.webp' # Update with your test image path

# The threshold for YOLO to even consider drawing a box (0.5 = 50% sure)
YOLO_CONF_THRESHOLD = 0.15 

# Must match your exact 7 folders (including 'healthy') in alphabetical order
CLASS_NAMES = [
    'Crown_Rot_Disease', 
    'Fruit_Fasciation_Disorder', 
    'Fruit_Rot_Disease', 
    'Mealybug_Wilt_Disease', 
    'Multiple_Crown_Disorder', 
    'No_Disease', 
    'Root_Rot_Disease'
]
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Load Models ---
yolo_model = YOLO(YOLO_MODEL_PATH)

effnet_model = models.efficientnet_b0(weights=None)
in_features = effnet_model.classifier[1].in_features
effnet_model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
effnet_model.load_state_dict(torch.load(EFFNET_MODEL_PATH, weights_only=True))
effnet_model = effnet_model.to(device)
effnet_model.eval()

effnet_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. Execute Pipeline ---
img = cv2.imread(TEST_IMAGE_PATH)

# Run YOLO, but we only extract the bounding box coordinates and its confidence score
results = yolo_model(img, verbose=False)[0]

for box in results.boxes:
    conf = box.conf[0].item()
    
    # Ignore the box if YOLO isn't confident
    if conf < YOLO_CONF_THRESHOLD:
        continue

    # DEBUG 1: Did YOLO find something?
    print(f"\n--- NEW DETECTION ---")
    print(f"YOLO spotted an anomaly (Confidence: {conf*100:.1f}%)")

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # 2. Crop the image
    crop = img[y1:y2, x1:x2]
    
    # DEBUG 2: Did the crop work?
    print(f"Crop shape extracted: {crop.shape}")
    
    if crop.size == 0:
        print("ERROR: Crop size is 0, skipping EfficientNet!")
        continue

    # 3. Format for EfficientNet
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    input_tensor = effnet_transforms(crop_pil).unsqueeze(0).to(device)

    # 4. EfficientNet Classification
    with torch.no_grad():
        outputs = effnet_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        final_class = CLASS_NAMES[predicted_idx.item()]
        effnet_conf_score = confidence.item()
        
        # DEBUG 3: What is EfficientNet thinking?
        print(f"EfficientNet classified it as: {final_class} ({effnet_conf_score*100:.1f}%)")

    # 5. Draw the final EfficientNet result on the image
    label = f"{final_class}: {effnet_conf_score*100:.1f}%"
    
    # Draw the bounding box (made it thicker so you can see it on the edges)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    
    # Calculate safe text placement
    text_y = y1 - 10 if y1 > 20 else y1 + 25 
    
    # Add a black background box for the text so it is always readable
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, text_y - text_height - 5), (x1 + text_width, text_y + 5), (0, 0, 0), -1)
    
    # Draw the green text over the black box
    cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the results
cv2.imshow("Option 1: Two-Stage Pipeline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()