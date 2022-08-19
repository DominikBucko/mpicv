from imutils.video import VideoStream
import argparse
import imutils
import cv2
import numpy as np
from mpi4py import MPI
from mtcnn import MTCNN
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def gaussian_on_pixel(frame, pix_coords_list, kernel_size=21):
    pixels = {}
    for pix_coords in pix_coords_list:
        pix_coords = (pix_coords[1], pix_coords[0])
        constraints = [0,0,0,0]
        constraints[0] = pix_coords[0] - int(kernel_size/2)
        constraints[1] = pix_coords[0] + int(kernel_size/2) + 1
        constraints[2] = pix_coords[1] - int(kernel_size/2)
        constraints[3] = pix_coords[1] + int(kernel_size/2) + 1

        for c in constraints:
            if c < 0:
                c = 0
            elif c > frame.shape[constraints.index(c) % 2]:
                c = frame.shape[constraints.index(c) % 2]

        frac = 1 / ((constraints[1] - constraints[0]) * (constraints[3] - constraints[2]))

        pix = [0, 0, 0]

        for x in range(constraints[0], constraints[1]):
            for y in range(constraints[2], constraints[3]):
                for chan in range(0, 3):
                    pix[chan] += frame[x][y][chan]

        for chan in range(0,3):
            pix[chan] = int(pix[chan] * frac)

        pixels[(pix_coords[0], pix_coords[1])] = pix
    return pixels

def greyscale_blur(frame):
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)


def movement_recognition(frame):
    gscale = greyscale_blur(frame)
    frameDelta = cv2.absdiff(firstFrame, gscale)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    for c in contours:
        if cv2.contourArea(c) < min_area_detection:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.drawContours(frame, [c], 0, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def face_detection_mtcnn(frame):
    boxes = face_detector.detect_faces(frame)
    if boxes:
        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        print(x,y,w,h)

        if do_blur:
            pixlist = []
            for x_c in range(x, x+w):
                for y_c in range(y, y+h):
                    pixlist.append((x_c, y_c))
            arrays = np.array_split(pixlist, max_threads)
            with ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                for arr in arrays:
                    futures.append(executor.submit(gaussian_on_pixel, frame, arr))
                for future in as_completed(futures):
                    blurred = future.result()
                    for key in blurred:
                        frame[key[0]][key[1]] = blurred[key]



        if conf > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    return frame


def capture_frame(stream):
    ret, cframe = stream.read()
    if ret:
        return cframe
    else:
        return None

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

max_threads = 4
dimensions = None
min_area_detection = 5000
firstFrame = None
stream = None
out = None
out_framerate = 10
face_detector = MTCNN()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video")
ap.add_argument("-f", "--face")
ap.add_argument("-r", "--rtsp")
ap.add_argument("-b", "--blur")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

do_blur = args.get("blur")
RTSP_ADDR = "rtsp://username:password@256.0.0.1:554/example

if rank == 0:
    if args.get("rtsp"):
        RTSP_ADDR = args.get("rtsp")
        stream = cv2.VideoCapture(RTSP_ADDR)
    elif args.get("video", None) is None:
        stream = VideoStream(src=0).start()
    else:
        stream = cv2.VideoCapture(args["video"])

    firstFrame = capture_frame(stream) if args.get("rtsp") or args.get("video") else stream.read()
    dimensions = firstFrame.shape
    out_framerate = 30
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))

dimensions = comm.bcast(dimensions, root=0)
#
if rank != 0:
    firstFrame = np.zeros(shape=dimensions, dtype=np.uint8)

comm.Bcast(firstFrame, root=0)
firstFrame = greyscale_blur(firstFrame)

while True:
    time_start = time.time()
    frames = None
    if rank == 0:
        if args.get("rtsp") or args.get("video"):
            frames = capture_frame(stream)
            if frames is None:
                break
            for i in range(0, world_size - 1):
                frame = capture_frame(stream)
                if frame is None:
                    frame = np.zeros(shape=dimensions, dtype=np.uint8)
                frames = np.append(frames, frame)
        else:
            frames = stream.read()
            for i in range(0, world_size-1):
                frame = stream.read()
                frames = np.append(frames, frame)

    currentFrame = np.zeros(shape=dimensions, dtype=np.uint8)

    comm.Scatter(frames, currentFrame, root=0)

    if args.get("face"):
        currentFrame = face_detection_mtcnn(currentFrame)
    else:
        currentFrame = movement_recognition(currentFrame)

    comm.Gather(currentFrame, frames, root=0)

    if rank == 0:
        frames = np.array_split(frames, world_size)
        for frame in frames:
            frame = frame.reshape(dimensions)

            out.write(frame)
            cv2.imshow("projekt", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

print("done")

MPI.Finalize()
exit(0)