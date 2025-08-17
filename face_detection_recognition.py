import cv2
import face_recognition
import os

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            encoding = face_recognition.face_encodings(image)
            if encoding:  #
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names

def recognize_faces(image_path, known_face_encodings, known_face_names, output_path="recognized_faces.jpg"):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    known_faces_dir = "known_faces"
    test_image_path = "test.jpg"
    output_path = "recognized_faces.jpg"

    
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    
    recognize_faces(test_image_path, known_face_encodings, known_face_names, output_path)