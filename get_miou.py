import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results
import numpy as np
'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   分类个数+1、如2+1
    #------------------------------#
    num_classes     = 21
    #--------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    #--------------------------------------------#
    name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")

        # 绘制 PA Recall 和 Precision 指标的折线图
        plt.figure(figsize=(12, 6))

        # PA Recall 折线图
        plt.subplot(1, 2, 1)
        for i in range(num_classes):
            plt.plot(np.arange(len(PA_Recall)), PA_Recall[i], label=name_classes[i])
        plt.title('PA Recall')
        plt.xlabel('Class')
        plt.ylabel('PA Recall')
        plt.legend(loc='best')

        # Precision 折线图
        plt.subplot(1, 2, 2)
        for i in range(num_classes):
            plt.plot(np.arange(len(Precision)), Precision[i], label=name_classes[i])
        plt.title('Precision')
        plt.xlabel('Class')
        plt.ylabel('Precision')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(miou_out_path, 'PA_Recall_Precision.png'))
        plt.show()