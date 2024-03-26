import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from face_detection import Face_utilities
from signalprocessing import Signal_processing
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import sys
from sklearn.decomposition import FastICA
from skimage import exposure
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import serial.tools.list_ports
import array


if __name__ == "__main__":

    port = serial.tools.list_ports.comports()
    serialInst = serial.Serial()

    prot_list = []

    for prot in port:
        prot_list.append(str(prot))
        print(str(prot))

    val = input("Select Port: COM")

    for x in range(0, len(prot_list)):
        if prot_list[x].startswith("COM" + str(val)):
            portVar = "COM" + str(val)
            print(portVar)

    serialInst.baudrate = 115200
    serialInst.port = portVar

    if not serialInst.isOpen():
        serialInst.open()
        print('com4 is open', serialInst.isOpen())

    path = input("option: ")

    if int(path) == 0:
        cap = cv2.VideoCapture(int(path))
        cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    else:
        cap = cv2.VideoCapture(str("r" + path))
        cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);

    fu = Face_utilities()
    sp = Signal_processing()

    i = 0
    last_rects = None
    last_shape = None
    last_age = None
    last_gender = None

    face_detect_on = False
    age_gender_on = False

    # for signal_processing
    BUFFER_SIZE = 256
    fps = 0  # for real time capture
    # R=[]
    # G=[]
    # B=[]=
    # data for plotting
    buffer_data = []
    # filtered_data = []
    # psd=[]
    # freqs_in_minute=[]
    bpm = 0
    bpms = []
    HR = []
    hr = 0


    # plotting
    # app = QtWidgets.QApplication([])
    # win = pg.GraphicsLayoutWidget(title="plotting")

    # p1 = win.addPlot(title="FFT")
    # p2 = win.addPlot(title ="Max30102")
    # p3 = win.addPlot(title = "remote_HR")
    # win.resize(640,480)
    def write_read(x):
        serialInst.write(bytes(x, 'utf-8'))


    def update():
        # p1.clear()
        # p1.plot(freqs_in_minute,psd, pen = 'g')

        p2.clear()
        p2.plot(HR, pen='g')

        p3.clear()
        p3.plot(bpms, pen='b')

        app.processEvents()


    # timer = QtCore.QTimer()
    # timer.timeout.connect(update)
    # timer.start(200)
    # win.show()
    # start_time=time.time()
    # print(start_time)
    # times.append(start_time)
    start_time = time.perf_counter()
    times = []
    num = input("Enter a number: ")
    value = write_read(num)

    while True:
        # t0 = time.time()
        if len(times) > 0:
            if times[-1] >= 60:
                cap.release()
                cv2.destroyAllWindows()
                break

        if (i % 1 == 0):
            face_detect_on = True
            if (i % 10 == 0):
                age_gender_on = True
            else:
                age_gender_on = False
        else:
            face_detect_on = False


        ret, frame = cap.read()
        times.append(time.perf_counter() - start_time)
        median_framerate = np.median(np.diff(times)) ** -1
        frame = cv2.flip(frame, 1)
        bb_box = frame.copy()
        cv2.putText(bb_box, "AVGfps: {0:.2f}".format(median_framerate), (30, int(frame.shape[0] * 0.95) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        if frame is None:
            print("End of video")
            cv2.destroyAllWindows()
            # timer.stop()

            break

        ret_process = fu.no_age_gender_face_process(frame)

        if ret_process is None:
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            # print(time.time()-t0)
            cv2.destroyAllWindows()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                # timer.stop()

                break
            continue

        rects, face, shape = ret_process

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])

        cv2.rectangle(bb_box, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(bb_box, "frame: {0:.2f}".format(i), (30, int(frame.shape[0] * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        img = frame
        bbbox = img.copy()
        ROI1 = img[shape[70][1]:shape[23][1], shape[70][0]:shape[23][0]]
        cv2.rectangle(bbbox, (shape[70][0], shape[70][1]), (shape[23][0], shape[23][1]), (0, 255, 0), 2)
        ROI2 = img[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]
        cv2.rectangle(bbbox, (shape[54][0], shape[29][1]), (shape[12][0], shape[33][1]), (0, 255, 0), 2)  # right cheeks
        ROI3 = img[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]
        cv2.rectangle(bbbox, (shape[4][0], shape[29][1]), (shape[48][0], shape[33][1]), (0, 255, 0), 2)  # left
        cv2.putText(bbbox, "frame: {0:.2f}".format(i), (30, int(frame.shape[0] * 0.95) - 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 0, 0), 2)

        Raw_rppg = sp.extract_color(ROI1,ROI2,ROI3)
        buffer_data.append(Raw_rppg)


        line = serialInst.readline().decode('utf-8')
        if len(line) < 10:
            hr = float(line)
            cv2.putText(bb_box, "HR: {0:.2f}".format(hr), (int(frame.shape[1] * 0.64), int(frame.shape[0] * 0.65)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            HR.append(hr)

        L = len(buffer_data)

        if L > BUFFER_SIZE:
            buffer_data = buffer_data[-BUFFER_SIZE:]
            # R=R[-BUFFER_SIZE:]
            # G= G[-BUFFER_SIZE:]
            # B=B[-BUFFER_SIZE:]
            times = times[-BUFFER_SIZE:]
            # bpms = bpms[-BUFFER_SIZE//2:]
            L = BUFFER_SIZE
        # print(times)

        if L == BUFFER_SIZE:
            diff = times[-1] - times[0]
            # print(diff)
            fps = float(L) / diff
            # cv2.putText(bb_box, "fps: {0:.2f}".format(fps), (30,int(frame.shape[0]*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            # RGB_array = np.array([R, G, B]).T
            # scaler = StandardScaler()
            # X_t = scaler.fit_transform(RGB_array) .round(4)

            # pca = PCA(n_components=3)
            # X_PCA = pca.fit_transform(X_t)

            # maxVinx=np.argmax(pca.explained_variance_ratio_)

            # signal=X_PCA[:,maxVinx]

            detrended_data = sp.signal_detrending(buffer_data)
            # interpolated_data = sp.interpolation(detrended_data, times)
            normalize_data = sp.normalization(detrended_data)
            filtered_data = sp.butter_bandpass_filter(normalize_data, 0.7, 4, fps, order=3)

            psd, freqs_in_minute, resolution = sp.fft(filtered_data, fps)
            idxfreq = np.where((freqs_in_minute > 45) & (freqs_in_minute < 180))[0]
            interest_idx_sub = idxfreq[:-1].copy()
            freqs_of_interest = freqs_in_minute[interest_idx_sub]
            fft_of_interest = psd[interest_idx_sub]
            max_arg = np.argmax(fft_of_interest)

            bpm = freqs_of_interest[max_arg]
            bpms.append(bpm)
            cv2.putText(bb_box, "HR: {0:.2f}".format(bpm), (int(frame.shape[1] * 0.64), int(frame.shape[0] * 0.95)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


        with open(r"C:\Users\User\Desktop\project report\realexp2.csv", mode="a+") as f:
            f.write("{0:.2f} ".format(times[-1]) + ", {0:.2f} ".format(hr) + ", {0:.2f} ".format(bpm) + "\n")


        cv2.imshow("frame", bb_box)
        i = i + 1

        # waitKey to show the frame and break loop whenever 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            # timer.stop()
            # sys.exit()
            break

    cap.release()
    cv2.destroyAllWindows()


