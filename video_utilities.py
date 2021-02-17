import cv2 as cv
import numpy as np

from file_utilities import save_descriptors

# capture = cv.VideoCapture("E:\\UNI\\VISIONE\\video\\Shelf_22.avi")
# if not capture.isOpened():
#     print('Unable to open: ' + args.input)
#     exit(0)

# frame_number = 0

# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break
    
#     frame_number = frame_number + 1

#     if (frame_number == 1107):
#         box = frame[69:(283 + 69), 0:177]
#         cv.imshow('Frame', box)
#         keyboard = cv.waitKey(30000)
#         if keyboard == 'q' or keyboard == 27:
#             break

videos = []
for x in range(1, 30):
    video_name = "Shelf_" + str(x) + ".avi"
    video = {"filename": video_name, "metadata": []}
    videos.append(video)

for x in range(1, 121):
    filename = "E:\\UNI\\VISIONE\\inSitu\\" + str(x) + "\\coordinates.txt"
    d = np.loadtxt("E:\\UNI\\VISIONE\\inSitu\\" + str(x) + "\\coordinates.txt", delimiter="\t")
    z = int(d[0][0]) # get video id
    d = d[:, 1:] # remove first column
    d = np.c_[d, np.full(d.shape[0], x)] #add a column containing the class ID
    videos[z - 1]["metadata"].append(d)

for v in videos:
    capture = cv.VideoCapture("E:\\UNI\\VISIONE\\video\\" + v["filename"])
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    frame_number = 0

    if len(v["metadata"]) > 0:

        stacked = np.vstack(v["metadata"])

        metadata = np.sort(stacked, axis = 0)

        next_frame_to_save = metadata[0][0]
        i = 0

        while True:
            ret, frame = capture.read()
            if frame is None:
                break        

            if (frame_number == next_frame_to_save):
                metadata_to_save = []
                while metadata[i][0] == next_frame_to_save and i < metadata.shape[0] - 1:
                    i = i + 1
                    metadata_to_save.append(metadata[i])
                if len(metadata_to_save) > 0:
                    cv.imwrite("test\\" + v["filename"][:-4] + "_frame_" + str(frame_number) + ".jpg", frame)
                    save_descriptors(metadata_to_save, "test\\" + v["filename"][:-4] + "_frame_" + str(frame_number))
                    next_frame_to_save = metadata[i][0]

            frame_number = frame_number + 1
