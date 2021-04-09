from torch import torch
import argparse

# tracking
from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.pipeline.utils import xywh2cxywh, get_crop, get_subwindow_tracking, tensor_to_numpy, xyxy2cxywh, cxywh2xywh
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from cv2 import cv2
import numpy as np
from PIL import Image
from trackingobj import trackthing


# multi Thread
from numba import config, njit, threading_layer, jit
import numba
from util import hyper_params, postprocess_box, postprocess_score, restrict_box
import threading
from tqdm import tqdm
from queue import Queue

import time

# classification
from knnmodel import init_KNN, get_data_from_video, init_get_video
from edge import offside_dectet
from ball_touch import ball_state

root_cfg.merge_from_file('./experiments/siamfcpp/siamfcpp_tinyconv.yaml')
task, task_cfg = specify_task(root_cfg)
task_cfg.freeze()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default= "./video/video5.mp4",#"rtmp://192.168.43.109:9999/live/test"
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
ap.add_argument("-d", "--detect", type=str, default="yolo",
                help="choose yolo or manul anchor at first")
args = vars(ap.parse_args())

classes_name = ["player", "ball", "team1", "team2", "judger"]
initBB = []
tracking_num = 50
process_length = 3
process_tracking_num = []
knn_numbers = [0, 0, 0]
process_to_insert = 0
knn_updated = False
knn_updating = False
knn_updating_number = None
mousex = 0
mousey = 0

fps = None

# Thread pool
config.THREADING_LAYER = 'threadsafe'
process = []
dataqueues = []
resultqueues = []

def get_point(event, x, y, flags, param):
    global knn_updated,knn_updating,mousex,mousey
    if event == cv2.EVENT_LBUTTONDOWN:
        if knn_updated == True:
            return 
        knn_updating = True
        mousex = x
        mousey = y

def multiprocessing_update(task, task_cfg, index, im, dataqueue, resultqueue):
    # build model
    Model = model_builder.build_model(task, task_cfg.model).to(torch.device("cuda"))
    Model.eval()
    target_pos = []
    target_sz = []
    im_z_crops = []
    lost = []
    features = []
    tracking_index = []
    total_num = 0
    avg_chans = np.mean(im, axis=(0, 1))
    im_h, im_w = im.shape[0], im.shape[1]
    z_size = hyper_params['z_size']
    x_size = hyper_params['x_size']
    context_amount = hyper_params['context_amount']
    phase = hyper_params['phase_init']
    phase_track = hyper_params['phase_track']
    score_size = (hyper_params['x_size'] -hyper_params['z_size']) // hyper_params['total_stride'] + 1 - hyper_params['num_conv3x3'] * 2
    if hyper_params['windowing'] == 'cosine':
        window = np.outer(np.hanning(score_size), np.hanning(score_size))
        window = window.reshape(-1)
    elif hyper_params['windowing'] == 'uniform':
        window = np.ones((score_size, score_size))
    else:
        window = np.ones((score_size, score_size))
    
    def init(state, im_x, total_num):
        for i in range(len(state)):
            target_pos.append(state[i][:2])
            target_sz.append(state[i][2:4])
            tracking_index.append(index*100+total_num+i)
            im_z_crop, _ = get_crop(im_x, target_pos[i], target_sz[i], z_size, avg_chans=avg_chans, context_amount=context_amount, func_get_subwindow=get_subwindow_tracking)
            im_z_crops.append(im_z_crop)
            array = torch.from_numpy(np.ascontiguousarray(im_z_crops[i].transpose(2, 0, 1)[np.newaxis, ...], np.float32)).to(torch.device("cuda"))
            lost.append(0)
            with torch.no_grad():
                features.append(Model(array,phase=phase))

    def delete_node(j):
        try: 
            del target_pos[j]
            del target_sz[j]
            del features[j]
            del tracking_index[j]
            del lost[j]
        except Exception as error:
            print("delete error",error)

    while True:
        try: 
            im_x, state, delete = dataqueue.get()
        except Exception as error:
            print(error)
            continue
        else:
            if len(state) > 0:
                init(state, im_x, total_num)
                total_num += len(state)
                continue
            if len(delete) > 0:
                delete_list = []
                for i in delete:
                    if i in tracking_index:
                        # print("delete",i)
                        node = tracking_index.index(i)
                        delete_node(node)
            
            result = []
            im = im_x.copy()
            del im_x, state, delete
            for i in range(len(features)):
                im_x_crop, scale_x = get_crop(im, target_pos[i], target_sz[i], z_size, x_size=x_size, avg_chans=avg_chans,context_amount=context_amount, func_get_subwindow=get_subwindow_tracking)
                array = torch.from_numpy(np.ascontiguousarray(im_x_crop.transpose(2, 0, 1)[np.newaxis, ...], np.float32)).to(torch.device("cuda"))
                with torch.no_grad():
                    score, box, cls, ctr, *args = Model(array, *features[i], phase=phase_track)
                
                box = tensor_to_numpy(box[0])
                score = tensor_to_numpy(score[0])[:, 0]
                cls = tensor_to_numpy(cls[0])
                ctr = tensor_to_numpy(ctr[0])
                box_wh = xyxy2cxywh(box)

                # #lost goal
                if score.max()<0.2:
                    lost[i] += 1
                    continue
                elif lost[i] > 0:
                    lost[i] -= 1
                best_pscore_id, pscore, penalty = postprocess_score(score, box_wh, target_sz[i], scale_x, window)
                # box post-processing
                new_target_pos, new_target_sz = postprocess_box(best_pscore_id, score, box_wh, target_pos[i], target_sz[i], scale_x,x_size, penalty)
                new_target_pos, new_target_sz = restrict_box(im_h, im_w, new_target_pos, new_target_sz)

                # save underlying state
                target_pos[i], target_sz[i] = new_target_pos, new_target_sz

                # return rect format
                track_rect = cxywh2xywh(np.concatenate([target_pos[i], target_sz[i]],axis=-1))
                result.append(track_rect)
            
            delete_list = []
            for i in range(len(features)):
                if lost[i] > 10:
                    delete_list = []
                    delete_list.append(i)
            for i in delete_list:
                delete_node(i)
            # print(index, len(features))
            resultqueue.put([result, tracking_index])   

def check(frame, result, yolo_objects, dataqueues):
    global process_to_insert, process_tracking_num
    temp = []
    for label, rect_box in result.items():
        if label in temp:
            continue
        for other_lable, other_box in result.items():
            if int(label) < int(other_lable) and rect_box.inbox_box(other_box.get_boxes()) > 0.9:
                if other_lable not in temp:
                    temp.append(other_lable)
                    print("there is a error")
    for label in temp:
        del result[label]
    if len(yolo_objects) == 0:
        return result, temp
    else:
        print(yolo_objects)
        add_object = [[]for i in range(process_length)]
        for objects in yolo_objects:
            if len(result) == tracking_num:
                break
            objects = np.append(cxywh2xywh(objects[0:4]),objects[-1])
            need = True
            if(objects[-1]<1):
                for rect_box in result.values():
                    IOU = rect_box.inbox_box(objects[0:4])
                    if IOU > 0.9:
                        need = False
                        break
            if need:
                # print("update",process_to_insert*100+process_tracking_num[process_to_insert], process_to_insert)
                result[str(process_to_insert*100+process_tracking_num[process_to_insert])] = trackthing(objects[0:4],objects[-1])
                add_object[process_to_insert].append(objects)
                process_tracking_num[process_to_insert] += 1
                process_to_insert = (process_to_insert+1)%process_length
        for i in range(process_length):
            dataqueues[i].put((frame, add_object[i], []))
        return result, temp

def initial_manuel(frame):
    initBB = []
    while True:
        initBB.append(cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True))
        cv2.rectangle(frame, (initBB[-1][0],initBB[-1][1]), (initBB[-1][0]+initBB[-1][2],initBB[-1][1]+initBB[-1][3]), (0, 255, 0), 2)
        initBB[-1] = xywh2cxywh(initBB[-1])
        key = cv2.waitKey(0) & 0xFF
        if key == ord("n"):
            initBB[-1] = np.append(initBB[-1],0)
        elif key == ord("q"):
            initBB = initBB[:-1]
            break
        else:
            initBB[-1] = np.append(initBB[-1],1)
    return initBB

def update_yolo(frame, yolo1):
    frame_clone = frame.copy()
    frame_clone = cv2.cvtColor(frame_clone,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame_clone = Image.fromarray(np.uint8(frame_clone))
    initBB = yolo1.detect_image_without_draw(frame_clone) 
    return initBB 

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('forkserver', force=True)
    # resolve config
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    init_get_video(classes_name[2:], "knn_classes")
    yolo_detect = 0
    yolo_update = False
    knn_classifier = None
    yolo_object = []
    delete = []
    tracking_object = dict()
    # loop over frames from the video stream
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    print(frame.shape[1], frame.shape[0])
    frame = frame[:,240:-240]
    cv2.namedWindow('Frame')
    frame = imutils.resize(frame, height=720, width=720)
    for i in range(process_length):
        dataqueues.append(torch.multiprocessing.Queue())
        resultqueues.append(torch.multiprocessing.Queue())
        process.append(torch.multiprocessing.Process(target=multiprocessing_update, args=(task, task_cfg, i, frame, dataqueues[-1], resultqueues[-1])))
        process[-1].start()
    if args["detect"] == "manual":
        initBB = initial_manuel(frame)     
                
    else:
        from yolo import YOLO
        yolo1 = YOLO()
        initBB = update_yolo(frame, yolo1) 
        initBB = initBB[:tracking_num]      

    for i in range(len(dataqueues)):
        temp = initBB[int(len(initBB)/len(dataqueues)*i):int(len(initBB)/len(dataqueues)*(i+1))]
        for j in range(len(temp)):
            tracking_object[str(i*100+j)] = trackthing(cxywh2xywh(temp[j][0:4]),temp[j][-1])
        process_tracking_num.append(len(temp))
        dataqueues[i].put((frame, temp, []))

    cv2.setMouseCallback('Frame',get_point)
    fps = FPS().start()
    while True:
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        frame = frame[:,240:-240]
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        frame = imutils.resize(frame, height=720, width=720)
        # frame = imutils.resize(frame, width=1000)
        frame_clone = frame.copy()
        if len(initBB)>0:
            if args["detect"] != "manuel":
                yolo_detect += 1
            if knn_updating:
                find = False
                for index, values in tracking_object.items():
                    if  values.inbox(mousex, mousey):
                        knn_updating_number = index
                        find = True
                        break
                if find:
                    (x, y, w, h) = [int(v) for v in tracking_object[knn_updating_number].get_boxes()]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255 , 255), 1)
                    cv2.imshow("Frame",frame)
                    print("add classes for this box")
                    key = cv2.waitKey(0) & 0xFF
                    print(key)
                    if key == ord(" "):
                        tracking_object[knn_updating_number].knn_update(2)
                    elif key == 13: #enter键
                        tracking_object[knn_updating_number].knn_update(3)
                    else:
                        tracking_object[knn_updating_number].knn_update(4)
                knn_updating = False
            for i in range(len(dataqueues)):
                dataqueues[i].put((frame, [], delete))
            delete = []
            for i in range(len(resultqueues)):
                try:
                    result, indexes = resultqueues[i].get(timeout=3)
                except RuntimeError:
                    print("lost")
                except Exception as error:
                    print("empty")
                else:
                    # print(indexes)
                    for i in range(len(result)):
                        label = result[i].copy()
                        try:
                            tracking_object[str(indexes[i])].update(label)
                        except KeyError:
                            continue
                    del result, indexes
            if yolo_update:
                yolo_object = update_yolo(frame, yolo1)
                yolo_detect = 0
                yolo_update = False
                
            tracking_object, delete = check(frame, tracking_object, yolo_object, dataqueues)
            yolo_object = []     
            if len(tracking_object) < tracking_num and yolo_detect > 100:
                yolo_update = True

            if knn_updated:
                ball_boxes = []
                player_boxes = []
            fuck_delete = [] 
            for i in tracking_object.keys():
                tracking_object[i].losted += 1
                if tracking_object[i].losted > 10:
                    fuck_delete.append(i)
                    continue
                if tracking_object[i].losted < 2 and knn_updated == False and tracking_object[i].knn_classes > 1 and knn_numbers[tracking_object[i].knn_classes-2]<25:
                    get_data_from_video(frame_clone, tracking_object[i].boxes, knn_numbers[tracking_object[i].knn_classes-2], classes_name[tracking_object[i].knn_classes], path="./knn_classes")
                    knn_numbers[tracking_object[i].knn_classes-2] += 1
                txt = int(tracking_object[i].get_class())
                (x, y, w, h) = [int(v) for v in tracking_object[i].get_boxes()]
                if knn_updated:
                    if txt == 0:
                        player_boxes.append([x,y,w,h])
                    elif txt == 1:
                        ball_boxes.append([x,y,w,h])
                if knn_updated and tracking_object[i].knn_classes == -1:
                    tracking_object[i].knn_update(knn_classifier.prediction(frame[int(y):int(y+h),int(x):int(x+w)]))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255 - txt*50, txt*50), 2)
                cv2.putText(frame, "{}".format(txt), (x + w//2, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255 - txt*50, txt*50), 1)
                cv2.putText(frame, i.format(txt), (x + w//2, y - h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255 - txt*50, txt*50), 1)
                tracking_object[i].updated = False
            if knn_updated:
                ball_boxes = np.array(ball_boxes)
                player_boxes = np.array(player_boxes)
                # state = 0:检测到大于一个球;1:触球;2:有球但未触球;3:球出边界所以没识别到;4:还未第一次检测到球;5:球在场内但没识别到
                state,touch_person = ball_state(frame, ball_boxes, player_boxes)

                #之后要通过touch_person判进攻or防守队伍
                if state == 2:  #之后要改成1
                    has_line, has_offside = offside_dectet(frame, 'up', touch_person[0], touch_person[1],player_boxes) #之后player_boxes要改成防守队伍的boxes
            # update the FPS counter

            for i in fuck_delete:
                del tracking_object[i]
            fps.update()
            fps.stop()
            if sum(knn_numbers)>=75:
                knn_updated = True
                knn_classifier = init_KNN("knn_classes")
            info = [
                ("Tracker", "siamfcpp"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, frame.shape[0] - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
    # close all windows
    cv2.destroyAllWindows()
    for i in range(len(process)):
        process[i].terminate()
        process[i].kill()
