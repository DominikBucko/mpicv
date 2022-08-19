from imutils.video import VideoStream
# import argparse
# import datetime
import imutils
# import time
# import cv2
# from mpi4py import MPI
# import numpy as np
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
# frame = np.empty(shape=(375, 500, 3), dtype="uint8")
# #
# if rank == 0:
#     vs = VideoStream(src=0).start()
# #     time.sleep(2.0)
# #
# #     firstFrame = None
# #
#     frame = vs.read()
# #     # text = "Unoccupied"
# #     # # resize the frame, convert it to grayscale, and blur it
# #     frame = imutils.resize(frame, width=500)
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.GaussianBlur(gray, (21, 21), 0)
# #     # if the first frame is None, initialize it
# #     if firstFrame is None:
# #         firstFrame = gray
# #     print("sending frame")
# #     data = np.zeros(shape=(377, 500, 3), dtype=np.uint8)
# #     data = 1
#
# #
# comm.Bcast(frame, root=0)
#
# # if rank != 0:
# #     print(frame)
# #     print("proc1")
#     # data = np.zeros(shape=(379, 500, 3), dtype=np.uint8)
# #     # data = None
# #
# #     print("receiving frame")
# #     buff = bytearray(20)
# #     data = comm.recv(source=0, tag=13)
# #     # cv2.imshow("Security Feed", data)
# #     print(data)
#

# from mpi4py import MPI
# import numpy as np
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
# if rank == 0:
#     data = np.zeros(shape=(375, 500, 3), dtype=np.uint8)
#
#     vs = VideoStream(src=0).start()
#     frame = vs.read()
#     frame = imutils.resize(frame, width=500)
#     data = (frame, 5)
#
# else:
#     data = (np.zeros(shape=(375, 500, 3), dtype=np.uint8), 0)
# comm.Bcast(data, root=0)
#
# if rank != 0:
#     print(data[1])
import cv2
input = cv2.VideoCapture("input.avi")
ret, frame = input.read()
dimensions = frame.shape
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))

while True:
    ret, frame = input.read()
    if not ret:
        break
    out.write(frame)
