import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools1 import generate_detections as gdet
from PIL import Image
from collections import deque
import velocity
import csv
out_csv = "position.csv"

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 516, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
pts = [deque(maxlen=3) for _ in range(9999)]
count = 1
Real_rate = 1   ## If the tracking accuracy is not 100%, the script counts some vehicles as double. 
                ## Therefore, the number of vehicles is greater than the actual number.
                ## This parameter compensates for the effect of tracking accuracy on counting accuracy.
                ## If tracking accuracy is 90%, set this value to 0.9.
def initialize_csv():
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicle_id", "pixel_x", "pixel_y"])

def main(_argv):
    initialize_csv()
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        # out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (1024, 768))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    velocity_tracker = velocity.VelocityTracker()
    number_of_frame_in_video = 0
    fps = 0.0
    T_count = 0
    go_count = 0
    come_count = 0
    tails = [] 
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        number_of_frame_in_video += 1

        img = cv2.resize(img, (1024, 768))

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            # cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            pts[int(track.track_id)].append(center[1])
            if len(pts[int(track.track_id)]) > 2 and abs(pts[int(track.track_id)][0] - pts[int(track.track_id)][2]) > 1:
                cv2.putText(img, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.6, color,2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                cv2.circle(img, (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2)), 2, (0, 0xFF, 0), thickness=2)
            ## Speed estimation and check entering status for counting.
            entered = velocity_tracker.entering_check(fps, number_of_frame_in_video, track.track_id, center, img, bbox)
            if entered == 1:
                if center[0] > 500:
                    go_count = go_count + 1
                else:
                    come_count = come_count + 1

            ## Write position per vehicle and frame
            pos_vicl_fram = [str(number_of_frame_in_video), str(track.track_id), str(center[0]), str(center[1])]
            with open(out_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(pos_vicl_fram)
            
        T_count = come_count * Real_rate + go_count * Real_rate
        # print count per frame on console.
        print("Frame: ", number_of_frame_in_video, "  ,  Count: ", nums)
        cv2.putText(img, "Frame: {:.2f}".format(number_of_frame_in_video), (0, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.putText(img, "Vehicles/Frame: {:.2f}".format(nums[0]), (0, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2) 
        # print count on screen
        cv2.putText(img, "Total Vehicles: {:.2f}".format(int(T_count)), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (5, 5, 255), 2) 
        cv2.putText(img, "Coming Vehicles: {:.2f}".format(int(come_count * Real_rate)), (500, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (5, 5, 255), 2) 
        cv2.putText(img, "Going Vehicles: {:.2f}".format(int(go_count * Real_rate)), (500, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (5, 5, 255), 2) 
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        if FLAGS.output:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

# python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi