from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2
import vlc


def apertura_occhio(occhio):
    #apertura dell'occhio in verticale
    A = dist.euclidean(occhio[1], occhio[5])
    B = dist.euclidean(occhio[2], occhio[4])

    #distanza tra i due estremi dell'occhio in orizzontale
    C = dist.euclidean(occhio[0], occhio[3])

    return ((A + B) / (2.0 * C))


SOGLIA_CHIUSURA = 0.28
SOGLIA_FRAME_CONSECUTIVI = 20
CONT = 0
ALLARME = False
p = vlc.MediaPlayer("sirena.mp3")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#LANDMARKS ID DEGLI OCCHI
(occhioSinistroInizio, occhioSinistroFine) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(occhioDestroInizio, occhioDestroFine) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vc = cv2.VideoCapture(0)

while True:
    ret, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        occhioSinistro = shape[occhioSinistroInizio:occhioSinistroFine]
        occhioDestro = shape[occhioDestroInizio:occhioDestroFine]
        aperturaOcchioSInistro = apertura_occhio(occhioSinistro)
        aperturaOcchioDestro = apertura_occhio(occhioDestro)
        aperturaMedia = (aperturaOcchioSInistro + aperturaOcchioDestro) / 2.0

        #DISEGNO I CONTRONI DEGLI OCCHI
        leftEyeHull = cv2.convexHull(occhioSinistro)
        rightEyeHull = cv2.convexHull(occhioDestro)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        cv2.putText(frame, "EAR: {:.2f}".format(aperturaMedia), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if aperturaMedia < SOGLIA_CHIUSURA:
            CONT += 1

            if CONT >= SOGLIA_FRAME_CONSECUTIVI:
                if not ALLARME:
                    ALLARME = True
                    p.play()

                cv2.putText(frame, "ATTENZIONE! SVEGLIA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            CONT = 0
            ALLARME = False
            p.stop()

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break