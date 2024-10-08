import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirpath")

pars = parser.parse_args()

cap = cv2.VideoCapture(pars.dirpath)

video_name = pars.dirpath.split("\\")[-1]
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    screen_width = frame.shape[1]
    screen_height = frame.shape[0]
    if not ret:
        break

    frame = cv2.resize(
        frame, (screen_width, screen_height), interpolation=cv2.INTER_AREA
    )

    text = f"Name: {video_name}, FPS: {fps:.2f}"

    cv2.putText(
        frame,
        text,
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Video", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
