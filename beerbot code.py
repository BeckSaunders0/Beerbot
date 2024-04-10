import cv2
import time
import mediapipe
from pyfirmata2 import Arduino, SERVO, util, OUTPUT
import math
import psycopg2

#Standard firmata example file should be uploaded onto Arduino 

#Face_Classifier = cv2.CascadeClassifier("C:\\Users\\flood\\OneDrive\\Desktop\\FaceTracking\\haarcascade_frontalface_default.xml")
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

cv2.namedWindow("Window")
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

port = 'COM4'
board = Arduino(port)
it = util.Iterator(board)
it.start()

pin1 = board.digital[9]
pin1.mode = SERVO
pin2 = board.digital[12]
pin2.mode = SERVO
pin3 = board.digital[8] #to solenoid valve
pin3.mode = OUTPUT

x_hat = 320 # [pixels]
y_hat = 240 # [pixels]
focal_length = 3.906 #inches 
known_distance = 3.85 #inches
distance = 60 #inches. Will adjust later to calculate on-the-fly 

def get_angle(x, x_hat):
    theta = math.degrees(math.atan(0.233 * (x - x_hat) / distance)) 
    return theta


with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        frame1 = cv2.resize(frame, (640, 480))

        results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks != None:
              for handLandmarks in results.multi_hand_landmarks:
                  drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)

                  for point in handsModule.HandLandmark:
                      
                    normalizedLandmark = handLandmarks.landmark[point]
                    pixelCoordinates_Landmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,normalizedLandmark.y, 640, 480)
                      
                    if point == 9: #middle finger bottom connection "center"
                        x = pixelCoordinates_Landmark[0]
                        y = pixelCoordinates_Landmark[1]

                        theta = get_angle(x, x_hat) + 90
                        pin1.write(theta)
                        time.sleep(.01)
                        phi = get_angle(y, y_hat) + 90
                        pin2.write(phi)

                        print(f"(Theta: {theta}, Phi: {phi})")

                        start_time = time.time()
                        if (time.time() - start_time < 2):
                            if ((x > 270 and x < 370) and (y > 190 and y < 290)): #Need to figure these out experimentally 
                                pin3.write(1)
                            else:
                                pin3.write(0)
                                continue

        cv2.imshow("frame", frame1)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break