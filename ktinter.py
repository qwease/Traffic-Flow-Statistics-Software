from datetime import datetime
import json
import os
import signal
import subprocess
from threading import Thread,Event
import threading

from tkinter import ttk
from tkinter import filedialog
# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import tkinter as tk
import tkinter.font as tkFont

import requests

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

#..... Tracker modules......
import skimage
from segment.sort_count import *
import numpy as np
#...........................
import math


FILE = Path(__file__).resolve()
ROOT = FILE.parent  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode


#............................... Tracker Functions ............................
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False

def orientation(p, q, r):
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
        return 1
    elif (val < 0):
        return 2
    else:
        return 0

def Intersection(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if ((o1 != o2) and (o3 != o4)):
        return True
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
    return False

def is_vehicle(det_cls):
    if (det_cls in ["car","truck","motorcycle","bus","train"]):# the train here means truck
        return True
    return False

def centroid_vector(centroids):
    #----------------------
    index1 = -1 
    index2 = -2
    #index 1 and 2 here defines how long is the gap to be determined
    if len(centroids) == 1:
        index2 = -1
    
    return  math.copysign(1,centroids[index2][1] - centroids[index1][1])*math.sqrt(pow(centroids[index2][0] - centroids[index1][0],2)+pow(centroids[index2][1] - centroids[index1][1],2))
#..............................................................................

class App:
    def __init__(self, window):
        #setting title
        window.title("浙江省高速车流量监控")
        window.iconbitmap("./favicon_64.ico")
        #setting window size
        self.width=630
        self.height=800
        screenwidth = window.winfo_screenwidth()
        screenheight = window.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (self.width, self.height, (screenwidth - self.width) / 2, (screenheight - self.height) / 2)
        window.geometry(alignstr)
        window.resizable(width=False, height=False)

        self.monitorLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.monitorLabel["font"] = ft
        self.monitorLabel["fg"] = "#333333"
        self.monitorLabel["justify"] = "center"
        self.monitorLabel["text"] = "监控图像"
        self.monitorLabel["relief"] = "flat"
        self.monitorLabel.place(x=280,y=290,width=58,height=30)

        streamHolderWidth=290
        self.streamHolder=Label(window)
        self.streamHolder["anchor"] = "center"
        # streamHolder["cursor"] = "watch"
        ft = tkFont.Font(family='Times',size=10)
        self.streamHolder["font"] = ft
        self.streamHolder["fg"] = "#333333"
        self.streamHolder["justify"] = "center"
        self.streamHolder["text"] = ""
        self.streamHolder["relief"] = "sunken"
        self.streamHolder.place(x=(self.width-streamHolderWidth)//2,y=10,width=streamHolderWidth,height=214)

        self.selectComboWidth=100
        self.selectCombo=ttk.Combobox(window)
        ft = tkFont.Font(family='Times',size=10)
        self.selectCombo["font"] = ft
        self.selectCombo.place(x=(self.width-self.selectComboWidth)//2,y=390,width=self.selectComboWidth,height=20) # "combox select: select from mainline, service area, hwts"

        self.cameraListComboWidth=200
        self.cameraListCombo=ttk.Combobox(window)
        ft = tkFont.Font(family='Times',size=10)
        self.cameraListCombo["font"] = ft
        self.cameraListCombo.place(x=(self.width-self.cameraListComboWidth)//2,y=420, width=self.cameraListComboWidth,height=20) # combox here: select camera
        


        self.startTimeLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.startTimeLabel["font"] = ft
        self.startTimeLabel["fg"] = "#333333"
        self.startTimeLabel["text"] = "开始时间"
        self.startTimeLabel.anchor=tk.W
        self.startTimeLabel.place(x=209,y=315,width=200,height=20)

        self.endTimeLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.endTimeLabel["font"] = ft
        self.endTimeLabel["fg"] = "#333333"
        self.endTimeLabel["text"] = "结束时间"
        self.endTimeLabel.anchor=tk.W
        self.endTimeLabel.place(x=209,y=340,width=200,height=20)

        self.durationLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.durationLabel["font"] = ft
        self.durationLabel["fg"] = "#333333"
        self.durationLabel["text"] = "监视时长"
        self.durationLabel.anchor=tk.W
        self.durationLabel.place(x=209,y=365,width=200,height=20)

        self.countingLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.countingLabel["font"] = ft
        self.countingLabel["fg"] = "#333333"
        self.countingLabel["justify"] = "center"
        self.countingLabel["text"] = "车辆计数"
        self.countingLabel.place(x=510,y=300,width=70,height=25)

        self.totalLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.totalLabel["font"] = ft
        self.totalLabel["fg"] = "#333333"
        self.totalLabel["justify"] = "center"
        self.totalLabel["text"] = "总计"
        self.totalLabel.place(x=480,y=340,width=70,height=25)

        self.carLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.carLabel["font"] = ft
        self.carLabel["fg"] = "#333333"
        self.carLabel["justify"] = "center"
        self.carLabel["text"] = "汽车"
        self.carLabel.place(x=560,y=340,width=70,height=25)

        self.truckLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.truckLabel["font"] = ft
        self.truckLabel["fg"] = "#333333"
        self.truckLabel["justify"] = "center"
        self.truckLabel["text"] = "卡车"
        self.truckLabel.place(x=480,y=390,width=70,height=25)

        self.busLabel=Label(window)
        ft = tkFont.Font(family='Times',size=10)
        self.busLabel["font"] = ft
        self.busLabel["fg"] = "#333333"
        self.busLabel["justify"] = "center"
        self.busLabel["text"] = "巴士"
        self.busLabel.place(x=560,y=390,width=70,height=25)

        self.saveResultCheckBoxVar=BooleanVar()
        self.saveResultCheckBox=Checkbutton(window,variable=self.saveResultCheckBoxVar)
        ft = tkFont.Font(family='Times',size=10)
        self.saveResultCheckBox["font"] = ft
        self.saveResultCheckBox["fg"] = "#333333"
        self.saveResultCheckBox["justify"] = "center"
        self.saveResultCheckBox["text"] = "保存监视结果"
        self.saveResultCheckBox.place(x=150,y=540,width=100,height=20)
        self.saveResultCheckBox["offvalue"] = "0"
        self.saveResultCheckBox["onvalue"] = "1"

        self.hideConfCheckBoxVar=BooleanVar()
        self.hideConfCheckBox=Checkbutton(window,variable=self.hideConfCheckBoxVar)
        ft = tkFont.Font(family='Times',size=10)
        self.hideConfCheckBox["font"] = ft
        self.hideConfCheckBox["fg"] = "#333333"
        self.hideConfCheckBox["justify"] = "center"
        self.hideConfCheckBox["text"] = "隐藏置信度"
        self.hideConfCheckBox.place(x=270,y=540,width=100,height=20)
        self.hideConfCheckBox["offvalue"] = "0"
        self.hideConfCheckBox["onvalue"] = "1"

        self.hideLabelsCheckBoxVar=BooleanVar()
        self.hideLabelsCheckBox=Checkbutton(window,variable=self.hideLabelsCheckBoxVar)
        ft = tkFont.Font(family='Times',size=10)
        self.hideLabelsCheckBox["font"] = ft
        self.hideLabelsCheckBox["fg"] = "#333333"
        self.hideLabelsCheckBox["justify"] = "center"
        self.hideLabelsCheckBox["text"] = "隐藏识别标签"
        self.hideLabelsCheckBox.place(x=380,y=540,width=100,height=20)
        self.hideLabelsCheckBox["offvalue"] = "0"
        self.hideLabelsCheckBox["onvalue"] = "1"

        self.monitorButtonWidth=70
        self.monitorButton=Button(window)
        self.monitorButton["bg"] = "#f0f0f0"
        # monitorButton["cursor"] = "watch"
        ft = tkFont.Font(family='Times',size=10)
        self.monitorButton["font"] = ft
        self.monitorButton["fg"] = "#000000"
        self.monitorButton["justify"] = "center"
        self.monitorButton["text"] = "开始监视"
        self.monitorButton.place(x=(self.width-self.monitorButtonWidth)//2,y=560,width=self.monitorButtonWidth,height=25)
        self.monitorButton["command"] = self.getURL

        self.terminateMonitorButtonWidth=70
        self.terminateMonitorButton=Button(window)
        self.terminateMonitorButton["bg"] = "#f0f0f0"
        # terminateMonitorButtonWidth["cursor"] = "watch"
        ft = tkFont.Font(family='Times',size=10)
        self.terminateMonitorButton["font"] = ft
        self.terminateMonitorButton["fg"] = "#000000"
        self.terminateMonitorButton["justify"] = "center"
        self.terminateMonitorButton["text"] = "停止监视"
        self.terminateMonitorButton["command"] = self.terminate_streaming

        consoleOutputWidth=self.width-30
        self.consoleOutput=Text(window)
        ft = tkFont.Font(family='Times',size=10)
        self.consoleOutput["font"] = ft
        self.consoleOutput["fg"] = "#333333"
        # self.consoleOutput["justify"] = "left"
        self.consoleOutput["relief"] = "sunken"
        
        self.outputScrollBar=Scrollbar(window)
        self.outputScrollBar.config(command=self.consoleOutput.yview)
        self.consoleOutput.config(yscrollcommand=self.outputScrollBar.set)
        self.consoleOutput.grid(column=0, row=0)
        self.outputScrollBar.grid(column=1, row=0, sticky='NS')
        self.consoleOutput.place(x=(self.width-consoleOutputWidth)//2,y=590,width=consoleOutputWidth,height=200)
        self.outputScrollBar.place(x=self.width-15,y=590,width=10,height=200)

        self.show_url = BooleanVar()
        self.URLRadio = Radiobutton(window, text="URL", variable=self.show_url, value=True, command=self.showUrl)
        self.localRadio = Radiobutton(window, text="本地文件", variable=self.show_url, value=False, command=self.showLocal)

        self.URLRadio.place(x=8,y=310,width=85,height=25)
        self.localRadio.place(x=20,y=380,width=85,height=25)

        #get camera list by types
        # type 1: 主线 https://ddzlcx.zjt.gov.cn/zlcx2/hikvisionVideo/videoList?onlineStatus=0&substreamUpStatus=1
        # type 2: 服务区 https://ddzlcx.zjt.gov.cn/zlcx2/vHwServiceArea/getHwServiceAreaList
        # type 3: 收费站 https://ddzlcx.zjt.gov.cn/zlcx2/hwTsStatus/getHwTsStatusList

        #主线的链接可以直接拿，但auth_key会变。
        #另外两个需要通过cameraCode拿。
        self.type_dict={}
        self.type_dict["高速公路主线"]="https://ddzlcx.zjt.gov.cn/zlcx2/hikvisionVideo/videoList?onlineStatus=0&substreamUpStatus=1"
        self.type_dict["服务区"]="https://ddzlcx.zjt.gov.cn/zlcx2/vHwServiceArea/getHwServiceAreaList"
        self.type_dict["收费站"]="https://ddzlcx.zjt.gov.cn/zlcx2/hwTsStatus/getHwTsStatusList"

        User_Agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        refresh_token = "30fcb494-7aa6-4cb8-9549-3f3c896a7cc9"
        authorization = "Bearer 5828ecc0-53d9-42f7-88a6-36bd1cf0555d"
        self.headers = {'Content-type': 'application/json;charset=UTF-8', 'Accept': 'application/json, text/plain, */*'}
        self.headers['User-Agent'] = User_Agent
        self.headers['Cookies']=f'authorization={authorization}; Hm_lvt_3a0d8bebfc9f336d035938249a76fcc9=1670829531; refreshAuth={refresh_token}; Hm_lpvt_3a0d8bebfc9f336d035938249a76fcc9=1670844435; SECKEY_ABVK=j/cUe1TzZH7bT2s/mFW0/Dy9u+/xNuVaf7plKH6AyXM%3D'
        self.headers["authorization"]="Bearer 5828ecc0-53d9-42f7-88a6-36bd1cf0555d"

        self.camList="./camList/"
        self.stop_threads = threading.Event()
        self.run_thread_pid=None
        self.show_url.set(True)

        self.text1 = Label(window, text='监控类型')
        self.text1.place(x=210, y=390, anchor="nw")
        self.text2 = Label(window, text='URL')
        self.text2.place(x=120, y=450, anchor='nw')

        self.cameras = ['自定义']
        self.urls = []
        self.camCodes=[]
        for camera in self.loadCamListFromText('./camList/mainLineList.txt'):
            self.cameras.append(camera['name'])
            self.camCodes.append(camera['cameraIndexCode'])
        self.cameraListCombo['value'] = self.cameras
        self.cameraListCombo.current(0)
        self.cameraListCombo.bind("<<ComboboxSelected>>", self.select)
        self.cameraListCombo.configure(state='readonly')
        
        self.selectCombo['value']=list(self.type_dict.keys())
        self.selectCombo.current(0)
        self.selectCombo.bind("<<ComboboxSelected>>", self.selectCamType)

        self.etext = Text(window, height=4)
        self.etext.place(x=(self.width-300)//2, y=450, width=300,anchor='nw')

        self.frm = Frame(window)
        self.frm.grid(padx='0', pady='0')
        
        self.entry = Entry(self.frm, width=40)
        self.entry.grid(row=0, column=0)
        self.uploadButton = Button(self.frm, text='上传文件', font=('Arial', 12), width=10, height=1, command=self.upload_file)
        self.uploadButton.grid(row=1, column=0, ipadx='3', ipady='3', padx='10', pady='20')
        self.frm.grid_forget()
    
    
    def showUrl(self):
        self.monitorButton.place(x=(self.width-self.monitorButtonWidth)//2,y=560,width=self.monitorButtonWidth,height=25)
        self.text1.place(x=210, y=390, anchor="nw")
        self.cameraListCombo.place(x=(self.width-self.cameraListComboWidth)//2,y=420, width=self.cameraListComboWidth,height=20)
        self.selectCombo.place(x=(self.width-self.selectComboWidth)//2,y=390,width=self.selectComboWidth,height=20)
        self.frm.grid_forget()
        self.frm.place_forget()
        if self.cameraListCombo.current() == 0:
            self.text2.place(x=120, y=450, anchor='nw')
            self.etext.place(x=(self.width-300)//2, y=450, width=300,anchor='nw')
        else:
            self.text2.place(x=180, y=420, anchor='nw')

    def showLocal(self):
        self.text1.place_forget()
        self.text2.place_forget()
        self.monitorButton.place(x=(self.width-self.monitorButtonWidth)//2,y=560,width=self.monitorButtonWidth,height=25)
        self.cameraListCombo.place_forget()
        self.selectCombo.place_forget()
        self.etext.place_forget()
        self.frm.grid(padx='0', pady='0')
        self.frm.place(x=175,y=400,width=300)

    def show_frames(self,im0):
        # Get the latest frame and convert into Image
        cv2image= im0
        img = Image.fromarray(cv2image)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image = img)
        self.streamHolder.imgtk = imgtk
        self.streamHolder.configure(image=imgtk)
        # Repeat after an interval to capture continiously
        self.streamHolder.after(int((1/25)*1000), self.show_frames,im0)

    # Define function to show frame
    def predict(self,url):
        self.predictURL=url
        self.driver()
        self.monitorButton.place_forget()
        self.terminateMonitorButton.place(x=(self.width-self.terminateMonitorButtonWidth)//2,y=560,width=self.terminateMonitorButtonWidth,height=25)
        self.camType=self.selectCombo.current()
        self.logToConsole("Start monitoring..."+"\n")

    def getURL(self):
        if self.show_url.get():
            if self.cameraListCombo.current() == 0:
                self.logToConsole("Getting user URL..."+"\n")
                URL = self.etext.get('1.0', 'end')
                while URL.endswith('\n'):
                    URL=URL.rstrip()
            else:
                self.logToConsole("Connecting remote URL..."+"\n")
                camCodes = self.camCodes[self.cameraListCombo.current() - 1]
                URL=self.getVideoUrlByCameraCode(camCodes)
                self.logToConsole("Got remote URL"+"\n")
        else:
            URL = self.entry.get()
            
        
        is_file = Path(URL).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url=URL.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url or is_file:
            self.logToConsole(self.cameras[self.cameraListCombo.current()]+": "+URL+"\n")
            self.predict(URL)
        else:
            self.logToConsole(f"\'{URL}\' 不是一个URL链接或一个文件地址"+"\n")

    def logToConsole(self,msg):
        self.consoleOutput.insert(END,msg)
        self.consoleOutput.see(END)

    def loadCamListFromText(self,txtURL):
        with open(txtURL, "r", encoding="utf-8") as f:
            camList=f.readlines()
            camList="".join(camList).replace('\'','"')
            camList=json.loads(camList)
        
        return camList

    def getVideoUrlByCameraCode(self,cameraCode,typeCode=2):
        getVideoUrlByCameraCode="https://ddzlcx.zjt.gov.cn/zlcx2/hikvisionVideo/getVideoUrlByCameraCode"

        HwTsVideoURL=requests.get(url=getVideoUrlByCameraCode,headers=self.headers,params={
                                'cameraCode' : cameraCode,
                                'type' : typeCode
                            })
        HwTsVideoURL=json.loads(HwTsVideoURL.text)['data']
        return HwTsVideoURL

    def select(self,event=None):
        index = self.cameraListCombo.current()  # 获取当前索引
        if self.cameraListCombo['value'][index] == '自定义':
            self.text2.place(x=120, y=450, anchor='nw')
            self.etext.place(x=(self.width-300)//2, y=450, width=300,anchor='nw')
        else:
            self.text2.place(x=180, y=420, anchor='nw')
            self.etext.place_forget()

    def selectCamType(self,event=None):
        self.camType=self.selectCombo.current() # integer
        filename=""
        self.cameras=['自定义']
        self.urls = []
        self.camCodes=[]
        if self.camType==0:
            # mainline
            filename="mainLineList.txt"
            for camera in self.loadCamListFromText(self.camList+filename):
                self.cameras.append(camera['name'])
                self.camCodes.append(camera['cameraIndexCode'])
            self.cameraListCombo['value'] = self.cameras
            self.cameraListCombo.current(0)
        elif self.camType==1:
            # 服务区
            filename="saList.txt"
            for sa in self.loadCamListFromText(self.camList+filename):
                for camera in sa['camList']:
                    self.cameras.append(camera['name'])
                    self.camCodes.append(camera['camCode'])
            self.cameraListCombo['value'] = self.cameras
            self.cameraListCombo.current(0)
        elif self.camType==2:
            # hwts
            filename="HwTsList.txt"
            for hwts in self.loadCamListFromText(self.camList+filename):
                for camera in hwts['camList']:
                    self.cameras.append(camera['name'])
                    self.camCodes.append(camera['camCode'])
            self.cameraListCombo['value'] = self.cameras
            self.cameraListCombo.current(0)
        self.select()

    def upload_file(self):
        selectFile = filedialog.askopenfilename(filetypes=(("MP4视频", "*.mp4"),))  # askopenfilename 1次上传1个；askopenfilenames1次上传多个
        self.entry.insert(0, selectFile)

    @smart_inference_mode()
    def run(self,
            weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/predict-seg',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=1,  # bounding box thickness (pixels)
            hide_labels=True,  # hide labels
            hide_conf=True,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            trk = False,
    ):  
        
        self.run_thread_pid=os.getpid()
        weights="./yolov7-seg.pt"
        nosave=not self.saveResultCheckBoxVar.get()
        hide_conf=self.hideConfCheckBoxVar.get()
        hide_labels=self.hideLabelsCheckBoxVar.get()
        source=self.predictURL
        trk=True

        self.logToConsole(
            str({'weights': weights, 'source': source, 'data': data, 'imgsz': imgsz, 'conf_thres': conf_thres, 'iou_thres': iou_thres, 'max_det': max_det, 'device': device, 'view_img': view_img, 'save_txt': save_txt, 'save_conf': save_conf, 'save_crop': save_crop, 'nosave': nosave, 'classes': classes, 'agnostic_nms': False, 'augment': False, 'visualize': False, 'update': False, 'project': project, 'name': name, 'exist_ok': exist_ok, 'line_thickness': line_thickness, 'hide_labels': hide_labels, 'hide_conf': hide_conf, 'half': half, 'dnn': dnn, 'trk': trk})
            +"\n"
            )

        self.carCount=0
        self.truckCount=0
        self.motorcycleCount=0
        self.busCount=0

        video_name=str(self.get_current_timestamp())
        #.... Initialize SORT .... 
            
        sort_max_age = 5 
        sort_min_hits = 2
        sort_iou_thresh = 0.2
        vehicle_up_count =0
        vehicle_down_count =0
        cross_length= 2.9 #determine how long shall the gap be to be counted as crossing
        # kalman filtering shall be applied for tracking
        seen_set={}
        sort_tracker = Sort(max_age=sort_max_age,
                            min_hits=sort_min_hits,
                            iou_threshold=sort_iou_thresh) 
        #......................... 

        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        project=ROOT/'runs'
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # print(names)
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size

            save_dir=project/"stream"
            self.logToConsole(f"Successfully load {source} at {dataset.fps[0]:.2f} FPS)"+"\n")
        else:
            view_img = check_imshow()
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
            save_dir=project/"video"
        
        self.startTime=self.get_current_timestamp()
        self.startTimeLabel["text"]="开始时间: "+str(datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,exist_ok=True)  # make dir
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        #gui things
        

        for path, im, im0s, vid_cap, s in dataset:

            if self.terminate==True:
                dataset.terminate_cap=True
            im0s=np.array(im0s)
            if len(im0s.shape)==4:
                height_img,width_img,ch_img = im0s.shape[1:]
            else:
                height_img,width_img,ch_img = im0s.shape
            
            max_size=max(height_img,width_img)
            if max_size>480:
                ratio=min(height_img,width_img)/max_size
                if height_img>=width_img:
                    resize_height_img=480
                    resize_width_img=int(resize_height_img*ratio)
                else:
                    resize_width_img=480
                    resize_height_img=int(resize_width_img*ratio)
            else:
                resize_width_img=width_img
                resize_height_img=height_img
            self.streamHolder.place(x=(self.width-resize_width_img)//2,y=10,width=resize_width_img,height=resize_height_img)
            #--------------------------------------
            up_line_pt_1 = (0, 13*height_img // 20)
            up_line_pt_2 = (width_img, 13*height_img // 20)
            up_line_pt_3 = (0, 8*height_img // 20)
            up_line_pt_4 = (width_img, 8*height_img // 20)

            down_line_pt_1 = (0, 12*height_img // 20)
            down_line_pt_2 = (width_img, 12*height_img // 20)
            down_line_pt_3 = (0, 9*height_img // 20)
            down_line_pt_4 = (width_img, 9*height_img // 20)
            # a line drawn at the middle
            #--------------------------------------
            
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, out = model(im, augment=augment, visualize=visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                save_path=str(p.parent/ (video_name+p.suffix))
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(6), True) for cls in det[:, 5]]
                    im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                    annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                    # Mask plotting ----------------------------------------------------------------------------------------

                    
                    det_cls=""
                    # Write results
                    for *xyxy, conf, cls in reversed(det[:, :6]):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            # print(names[c])
                            det_cls=names[c]
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    if trk:
                        #Tracking ----------------------------------------------------
                        dets_to_sort = np.empty((0,6))
                        for x1,y1,x2,y2,conf,detclass in det[:, :6].cpu().detach().numpy():
                            dets_to_sort = np.vstack((dets_to_sort, 
                                            np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))

                        tracked_dets = sort_tracker.update(dets_to_sort)
                        tracks =sort_tracker.getTrackers()


                        for track in tracks:#different object is assigned with different track
                            seen_set[track.id]=False
                            #----------------------
                            index1 = -1 
                            index2 = -2
                            #index 1 and 2 here defines how long is the gap to be determined
                            if len(track.centroids) == 1:
                                index2 = -1
                            
                            up_interseted_1 = Intersection(
                                                    Point(track.centroids[index1][0], track.centroids[index1][1]),
                                                    Point(track.centroids[index2][0], track.centroids[index2][1]),
                                                    Point(up_line_pt_1[0],up_line_pt_1[1]), 
                                                    Point(up_line_pt_2[0], up_line_pt_2[1])
                                                    )
                            
                            up_interseted_2= True if up_interseted_1 else Intersection(
                                                    Point(track.centroids[index1][0], track.centroids[index1][1]),
                                                    Point(track.centroids[index2][0], track.centroids[index2][1]),
                                                    Point(up_line_pt_3[0],up_line_pt_3[1]), 
                                                    Point(up_line_pt_4[0], up_line_pt_4[1])
                                                    )

                            down_interseted_1 = Intersection(
                                                    Point(track.centroids[index1][0], track.centroids[index1][1]),
                                                    Point(track.centroids[index2][0], track.centroids[index2][1]),
                                                    Point(down_line_pt_1[0],down_line_pt_1[1]), 
                                                    Point(down_line_pt_2[0], down_line_pt_2[1])
                                                    )

                            down_interseted_2 = True if down_interseted_1 else Intersection(
                                                    Point(track.centroids[index1][0], track.centroids[index1][1]),
                                                    Point(track.centroids[index2][0], track.centroids[index2][1]),
                                                    Point(down_line_pt_3[0],down_line_pt_4[1]), 
                                                    Point(down_line_pt_3[0], down_line_pt_4[1])
                                                    )
                            if is_vehicle(det_cls):
                                if centroid_vector(track.centroids)>cross_length and (up_interseted_1 == True) and not seen_set[track.id]:
                                    vehicle_up_count+=1
                                    if det_cls =="car":
                                        self.carCount+=1
                                    elif det_cls=="truck" or det_cls=="train":
                                        self.truckCount+=1
                                    elif det_cls=="motorcycle":
                                        self.motorcycleCount+=1
                                    elif det_cls=="bus":
                                        self.busCount+=1
                                elif centroid_vector(track.centroids)<-cross_length and (down_interseted_1==True) and not seen_set[track.id]:
                                    vehicle_down_count+=1
                                    if det_cls =="car":
                                        self.carCount+=1
                                    elif det_cls=="truck" or det_cls=="train":
                                        self.truckCount+=1
                                    elif det_cls=="motorcycle":
                                        self.motorcycleCount+=1
                                    elif det_cls=="bus":
                                        self.busCount+=1
                                
                                
                                self.totalLabel["text"]="总计: "+str(vehicle_up_count+vehicle_down_count)
                                self.carLabel["text"]="汽车: " + str(self.carCount)
                                self.busLabel["text"]="巴士: " + str(self.busCount)
                                self.truckLabel["text"]="卡车: "+ str(self.truckCount)

                            #----------------------
                            annotator.draw_trk(line_thickness,track)
                            seen_set[track.id]=True
                        if len(tracked_dets)>0:
                            bbox_xyxy = tracked_dets[:,:4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            identities = tracked_dets[:, 8]
                            # for id in identities:
                            #     seen_set[id]=True
                            annotator.draw_id(bbox_xyxy, identities, categories, names)
                annotator.draw_counting(up_line_pt_1,up_line_pt_2,
                                        down_line_pt_1,down_line_pt_2,
                                        width_img,height_img,
                                        vehicle_up_count,vehicle_down_count)

                # Stream results
                im0 = annotator.result()

                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    # cv2.imshow(str(p), im0)
                    # self.show_frames(im0)
                    img_resize=im0.copy()
                    img_resize = cv2.resize(img_resize, (resize_width_img,resize_height_img), interpolation = cv2.INTER_AREA)
                    img = Image.fromarray(img_resize)
                    # Convert image to PhotoImage
                    imgtk = ImageTk.PhotoImage(image = img)
                    self.streamHolder.configure(image=imgtk)
                    # self.streamHolder['image']=imgtk
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if dataset.mode == 'stream':
                            save_path=str(Path(project/"stream"/video_name).with_suffix('.mp4'))
                        else:
                            save_path=str(Path(project/"video"/video_name).with_suffix('.mp4'))
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            self.logToConsole(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"+"\n")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        self.logToConsole(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t+"\n")
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_path)}{s}"+"\n")
            self.logToConsole(f"Results saved to {colorstr('bold', save_path)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        self.terminate=True
        self.terminate_streaming()

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--trk', action='store_true', help='Apply Sort Tracking')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        
        
        # print(type(vars(opt)))#dict
        # print(type(opt))
        
        return opt

    def driver(self):
        opt = self.parse_opt()
        check_requirements(exclude=('tensorboard', 'thop'))
        arg_list=[]
        for arg in vars(opt).values():
            arg_list.append(arg)
        # print(arg_list)
        self.terminate=False
        self.stop_threads.clear()
        self.thread = Thread(target = self.run, args = arg_list)
        self.thread.start()

        self.logToConsole("Starting thread..."+"\n")

        self.startTimeLabel["text"]="开始时间"
        self.endTimeLabel["text"]="结束时间"
        self.durationLabel["text"]="监视时长"

    def terminate_streaming(self):
        self.terminate=True
        # self.stop_threads.set()
        # os.kill(self.run_thread_pid, signal.SIGINT)
        # self.thread.join()
        self.thread=None
        self.terminateMonitorButton.place_forget()
        self.monitorButton.place(x=(self.width-self.terminateMonitorButtonWidth)//2,y=560,width=self.terminateMonitorButtonWidth,height=25)

        self.endTime=self.get_current_timestamp()
        
        self.endTimeLabel["text"]="结束时间: "+str(datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
        self.duration=self.endTime-self.startTime
        self.durationLabel["text"]="监视时长: "+str(int(self.duration/1000))+"秒"

    def get_current_timestamp(self):
        # in the website usable format: e.g. 1670831357971
        return int(math.floor(time.time()*1000))

if __name__ == "__main__":
    window = Tk() # https://visualtk.com/
    app = App(window)
    this_pid=os.getpid()
    window.mainloop()
    from numba import cuda
    for device_index in range(torch.cuda.device_count()):
        cuda.select_device(device_index)
        cuda.close()
    os.kill(this_pid,signal.SIGINT)
    

# record_json: camCode, start time, end time, counting, cars