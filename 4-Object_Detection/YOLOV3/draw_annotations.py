import numpy as np
import glob
import cv2
import os
import time

from PIL import Image
import shutil

    
def openfilebox(bbox_txt):
    _list=[]
    if os.path.isfile(bbox_txt):
        with open(bbox_txt,'r') as f:
            _lines = f.readlines()
            for line in _lines:
                _str = line.rstrip('\n')
                if len(_str)>3:
                    temp = _str.split(' ')
                    temp = [i for i in temp if len(i)>0]
                    # print("{} -> {}".format(temp[0],temp[2]))
                    temp.remove(temp[0])
                    _list.append(temp)    
    return _list



def swap_value(val1,val2):
    temp = val1
    val1 = val2
    val2 = temp
    return val1, val2


def drawOnImgs(path_imgs, dir_groundtruth, dir_predict, path_output,draw_text=True):

    if not os.path.exists(path_imgs):
        print("error dont exist %"%path_imgs)
        # shutil.rmtree(predicted_dir_path)
    if os.path.exists(path_output):
        # print("error dont exist %"%path_imgs)
        shutil.rmtree(path_output)
    os.mkdir(path_output)
    

    imgs = os.listdir(path_imgs)
    gts = os.listdir(dir_groundtruth)
    dts = os.listdir(dir_predict)
    
    imgs = sorted(imgs)
    gts = sorted(gts)
    dts = sorted(dts)

    if len(imgs)!=len(gts) or len(imgs)!=len(dts):
        print("error size %s %s %s"%path_imgs, dir_groundtruth, dir_predict)
        return

    for img,gt,dt in zip(imgs,gts,dts):
        image = cv2.imread(os.path.join(path_imgs,img))
        _list_gt = openfilebox(os.path.join(dir_groundtruth,gt))
        _list_dt = openfilebox(os.path.join(dir_predict,dt))
        
        print('--------1')
        print(_list_gt)
        print('--------2')
        print(_list_dt)
        print('--------3')

        if len(_list_gt)>0:
            for coor in _list_gt:
                # bbox_color_gt = (0, 255, 0) # verde (BGR)
                bbox_color_gt = (204, 255, 0) # verde (BGR)
                bbox_thick_gt = 2
                text_thickness_gt = 0.6
                # c1_gt, c2_gt = (int(float(coor[0])), int(float(coor[1]))), (int(float(coor[2])), int(float(coor[3])))
                coor1, coor2, coor3, coor4 = int(float(coor [0])), int(float(coor[1])), int(float(coor[2])), int(float(coor[3]))
                # print("gt-----1")
                # print(coor1, coor2, coor3, coor4)
                # print("gt-----2")
                c1_gt, c2_gt = ((coor1, coor2),(coor3, coor4))
                cv2.rectangle(image, c1_gt, c2_gt, bbox_color_gt, bbox_thick_gt)
                if draw_text:
                    # namelabel = 'groundtruth' 
                    namelabel = 'gt' 
                    # cv2.putText(image, namelabel, (coor1, coor2-10), cv2.FONT_HERSHEY_SIMPLEX, text_thickness_gt, bbox_color_gt, 2)
                    cv2.putText(image, namelabel, (coor1, coor2), cv2.FONT_HERSHEY_SIMPLEX, text_thickness_gt, bbox_color_gt, 2)


        # if len(_list_dt)>0:
        #     for coor in _list_dt:
        #         bbox_color_dt = (255, 0, 0) #verde (BGR)
        #         bbox_thick_dt = 2
        #         text_thickness_dt = 0.6
        #         # c1_dt, c2_dt = (int(float(coor[1])), int(float(coor[2]))), (int(float(coor[3])), int(float(coor[4])))
        #         coor1, coor2, coor3, coor4 = int(float(coor[1])), int(float(coor[2])), int(float(coor[3])), int(float(coor[4]))
        #         # print('dt.....1')
        #         # print(coor1, coor2, coor3, coor4)
        #         # print('dt.....2')
        #         c1_dt, c2_dt = ((coor1, coor2),(coor3, coor4))
        #         # c1_dt, c2_dt = (int(coor[1]), int(coor[2])), (int(coor[3]), int(coor[4]))
        #         cv2.rectangle(image, c1_dt, c2_dt, bbox_color_dt, bbox_thick_dt)
        #         if draw_text:
        #             # namelabel = 'detection'
        #             namelabel = 'dt'
        #             # cv2.putText(image, namelabel, (coor1, coor2-10), cv2.FONT_HERSHEY_SIMPLEX, text_thickness_dt, bbox_color_dt, 2)
        #             cv2.putText(image, namelabel, (coor1, coor2), cv2.FONT_HERSHEY_SIMPLEX, text_thickness_dt, bbox_color_dt, 2)

        cv2.imwrite(os.path.join(path_output,img), image) 




if __name__ == '__main__':
    

    groundtruth_dir_path = './results/ground-truth'
    predicted_dir_path = './results/predicted'
    imgs_dir_path = './results/imgs'

    drawBB_dir_path = './results/drawBB_imgs'


    drawOnImgs(imgs_dir_path, groundtruth_dir_path, predicted_dir_path, drawBB_dir_path)
