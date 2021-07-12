import cv2

from utils.prediction import predict_letter

cap = cv2.VideoCapture(-1)

while True:
    ret, frame = cap.read()
    # draw rectangle
    cv2.rectangle(frame, (90, 50), (540, 450), (255, 0, 0), 4)

    # Region of interest
    roi = frame[50:450, 90:540]

    # convert to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # resize ROI into 28x28
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imshow('roi', roi)

    result = predict_letter(roi)

    cv2.putText(
        frame, result, (300, 100),
        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0),
        2)
    cv2.imshow('frame', frame)

    # close by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
