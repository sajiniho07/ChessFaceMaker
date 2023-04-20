import cv2
import numpy as np

class FaceDetector():
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier("res/haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)

    def filter_chess_image(self, img, width):
        kernel = cv2.getGaussianKernel(width, 10)
        kernel = np.outer(kernel, kernel.transpose())
        return cv2.filter2D(img, -1, kernel)

    def stream_webcam(self):
        show_filter = True
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame =  cv2.flip(frame, 1)
                desc = "Control chess face with 'C' button."
                BLACK = (0, 0, 0)
                cv2.putText(frame, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    show_filter = not show_filter

                if show_filter:
                    self.set_chess_filter_on_face(frame)

                cv2.imshow("temp_frame", frame)
                if key == ord('q'):
                    break

        cv2.destroyAllWindows()
        self.cap.release()

    def set_chess_filter_on_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_results = self.face_detector.detectMultiScale(gray)
        for(x, y, w, h) in face_results:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            sub_img = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = self.filter_chess_image(sub_img, w)


mp = FaceDetector()
mp.stream_webcam()