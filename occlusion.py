import cv2
import dlib
import numpy as np

class Img:
    def __init__(self, url, x1, x2, y1, y2):
        self.url = url
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

def extract_frame_number(frame_url):
    # Extract the frame number from the filename using regex
    match = re.search(r"frame_(\d+)\.jpg", os.path.basename(frame_url))
    if match:
        frame_number = int(match.group(1))
        prefix = os.path.dirname(frame_url)
        return prefix, frame_number
    return None, None

def construct_frame_url(prefix, frame_number):
    return os.path.join(prefix, f"frame_{frame_number:05d}.jpg")

def load_tracks(file_path):
    with open(file_path, 'rb') as f:
        tracks = pickle.load(f)
    print(f"Tracks loaded from {file_path}")
    return tracks

# Load Dlib's pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load pre-trained YOLO object detector
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def read_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
    return images

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], class_ids[i]) for i in range(len(boxes)) if i in indexes]

def detect_landmarks(frame, face_rect):
    landmarks = predictor(frame, face_rect)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    return landmarks_points

def detect_occlusion(images):
    occlusions = []
    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        occluded = False
        closest_distance = float('inf')

        for face in faces:
            landmarks = detect_landmarks(gray, face)
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            objects = detect_objects(frame)
            for (box, class_id) in objects:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if x < face.right() and x + w > face.left() and y < face.bottom() and y + h > face.top():
                    occluded = True
                    distance = min([np.linalg.norm(np.array([x - lx, y - ly])) for lx, ly in landmarks])
                    closest_distance = min(closest_distance, distance)

        occlusions.append((frame, occluded, closest_distance))
    return occlusions


images = read_images_from_directory("processed-videos/Zendaya Talks Euphoria Season 2, Her Iconic Looks, & Spider-Man ｜ Fan Mail ｜ InStyle/frames")
results = detect_occlusion(images)

for i, (frame, occluded, distance) in enumerate(results):
    if occluded:
        print(f"Frame {i}: Face occluded. Closest object distance: {distance}")
    # else:
    #     print(f"Frame {i}: No occlusion detected")
    # cv2.imshow(f"Frame {i}", frame)
