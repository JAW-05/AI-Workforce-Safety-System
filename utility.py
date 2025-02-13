import openvino as ov
import cv2
import numpy as np

core = ov.Core()

# Load models
model_face = core.read_model(model='models/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model=model_face, device_name="CPU")

model_emo = core.read_model(model='models/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")

model_ag = core.read_model(model='models/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model=model_ag, device_name="CPU")

# Get input and output layers
input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output(0)

def preprocess(image, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image

def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]

    # Filter boxes by confidence threshold
    mask = scores >= confidence_threshold
    scores = scores[mask]
    boxes = boxes[mask]

    # Ensure boxes are within valid ranges
    image_h, image_w, _ = image.shape
    boxes = np.clip(boxes, 0, [image_w, image_h, image_w, image_h])

    # Filter out boxes with invalid values
    valid_mask = np.all(np.isfinite(boxes), axis=1) & np.all(boxes >= 0, axis=1)
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]

    face_boxes = boxes.astype(np.int64)

    # Log the face boxes for debugging
    for i, box in enumerate(face_boxes):
        print(f"Face {i}: {box}, Score: {scores[i]}")

    return face_boxes, scores

def draw_age_gender_emotion(face_boxes, image):
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    show_image = image.copy()

    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]

        if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
            print(f"Invalid facebox coordinates: {face_boxes[i]}")
            continue

        face = image[ymin:ymax, xmin:xmax]

        if face.size == 0 or len(face.shape) != 3 or face.shape[2] != 3:
            print(f"Skipping empty or invalid face image at index {i}.")
            continue

        try:
            input_image = preprocess(face, input_layer_emo)
            results_emo = compiled_model_emo([input_image])[output_layer_emo]
            results_emo = results_emo.squeeze()
            index = np.argmax(results_emo)

            input_image_ag = preprocess(face, input_layer_ag)
            results_ag = compiled_model_ag([input_image_ag])
            age = int(np.squeeze(results_ag[1]) * 100)
            gender = np.squeeze(results_ag[0])
            gender_str = "female" if gender[0] > 0.65 else "male" if gender[1] >= 0.55 else "unknown"
            box_color = (200, 200, 0) if gender_str == "female" else (0, 200, 200) if gender_str == "male" else (200, 200, 200)

            font_scale = image.shape[1] / 750
            text = f"{gender_str} {age} {EMOTION_NAMES[index]}"
            print(f"Drawing box for {text} at {xmin}, {ymin}, {xmax}, {ymax}")
            cv2.putText(show_image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), box_color, 2)

        except Exception as e:
            print(f"Error processing face at index {i}: {e}")

    return show_image

def predict_image(image, conf_threshold):
    try:
        input_image = preprocess(image, input_layer_face)
        results = compiled_model_face([input_image])[output_layer_face]
        face_boxes, scores = find_faceboxes(image, results, conf_threshold)
        if len(face_boxes) == 0:
            print("No face boxes found.")
        else:
            print(f"Found {len(face_boxes)} face boxes.")
        visualize_image = draw_age_gender_emotion(face_boxes, image)
        return visualize_image
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return image
