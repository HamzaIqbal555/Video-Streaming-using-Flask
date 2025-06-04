from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(
    'models/age_deploy.prototxt',
    'models/age_net.caffemodel'
)

# Age categories in model
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Open webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Camera could not be opened.")


def generate_frame():
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603,
                                                                         87.7689143744, 114.895847746), swapRB=False)

                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_BUCKETS[age_preds[0].argmax()]
                label = f'Age: {age}'

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error during streaming: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
