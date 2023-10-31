import cv2
import sys
import numpy as np
import time

NUM_CLUSTERS = 3
NUM_ERROR_BITS = 8
MATCH_THRESH = 25
CLASSIFICATION_THRESH = 0.6
PIXEL_STEP = 2
DOWNSAMPLE_RATE = 2  # Process every 10th frame
INTERPOLATION_THRESHOLD = 2000  # You can adjust this threshold as needed

class Cluster:
    def __init__(self, weight=0.01, red=0xFF, green=0xFF, blue=0xFF):
        self.weight = weight
        self.red = red
        self.green = green
        self.blue = blue

    def __str__(self):
        return f'{self.weight=}, {self.red=}, {self.green=}, {self.blue=}'


video = cv2.VideoCapture('test4.mp4')
if not video.isOpened():
    print('Could not open video!')
    sys.exit()
start = False
count = 0

t = 0  # Initialize t for counting processed pixels

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        sys.exit()

    for _ in range(DOWNSAMPLE_RATE - 1):
        good, frame = video.read()
        if not good:
            print('Could not open video!')
            sys.exit()

    good, frame = video.read()
    if not good:
        print('Could not open video!')
        sys.exit()

    if not start:
        VIDEO_HEIGHT = frame.shape[0]
        VIDEO_WIDTH = frame.shape[1]
        #print(VIDEO_HEIGHT * VIDEO_WIDTH)
        groups = [[[Cluster() for _ in range(NUM_CLUSTERS)] for _ in range(VIDEO_WIDTH)] for _ in range(VIDEO_HEIGHT)]
        start = True

    processed = np.zeros_like(frame)

    start_time = time.time()

    for i in range(0, VIDEO_HEIGHT):
        for j in range(0, VIDEO_WIDTH):
            t += 1  # Increment t for each processed pixel
            red = frame[i][j][2]
            green = frame[i][j][1]
            blue = frame[i][j][0]

            match = None
            index = None
            for ind, cluster in enumerate(reversed(groups[i][j])):
                dist = abs(int(cluster.red) - int(red)) + abs(int(cluster.green) - int(green)) + abs(int(cluster.blue) - int(blue))
                if dist <= MATCH_THRESH:
                    match = cluster
                    index = NUM_CLUSTERS - 1 - ind
                    break

            if match is None:
                groups[i][j][0] = Cluster(weight=0.01, red=red, green=green, blue=blue)
                index = 0
            else:
                match.red += (int(red) - int(match.red)) >> 1
                match.green += (int(green) - int(match.green)) >> 1
                match.blue += (int(blue) - int(match.blue)) >> 1

                for ind, cluster in enumerate(groups[i][j]):
                    if ind == index:
                        cluster.weight += (1 / 35) * (1 - cluster.weight)  # L = 35
                    else:
                        cluster.weight += (1 / 35) * (0 - cluster.weight)

                total = 0
                for cluster in groups[i][j]:
                    total += cluster.weight
                for cluster in groups[i][j]:
                    cluster.weight /= total

                groups[i][j].sort(key=lambda c: c.weight)

                index = None
                for ind, cluster in enumerate(groups[i][j]):
                    if cluster is match:
                        index = ind

            total = 0
            for k in range(index + 1, NUM_CLUSTERS):
                total += groups[i][j][k].weight

            if total >= CLASSIFICATION_THRESH:
                processed[i][j] = 255
            else:
                processed[i][j] = 0
            
    # for i in range(0, VIDEO_HEIGHT):
    #     for j in range(0, VIDEO_WIDTH):
    #         # Only process pixels that were not initially processed
    #         top = np.sum(processed[i - 1, j]) if i > 0 else 0
    #         left = np.sum(processed[i, j - 1]) if j > 0 else 0
    #         right = np.sum(processed[i, j + 1]) if j < VIDEO_WIDTH - 1 else 0
    #         bottom = np.sum(processed[i + 1, j]) if i < VIDEO_HEIGHT - 1 else 0

    #         # Check if the majority of neighbors are white (255)
    #         white_neighbors = np.sum([top, left, right, bottom])
    #         if white_neighbors >= 600:  # Assuming 255 represents white
    #             interpolated_value = (255, 0, 0)  # Set the pixel to white
    #         else:
    #             interpolated_value = (0,255,0)  # Set the pixel to black

            # processed[i, j] = interpolated_value


    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Frame {count}: Processing time = {processing_time:.4f} seconds")
    print(f"Total Processed Pixels: {t}")

    cv2.imshow('project', frame)
    cv2.imshow('output', processed)
    count += 1
    t = 0
