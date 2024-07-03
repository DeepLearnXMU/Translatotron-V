import easyocr
import os
from tqdm import tqdm
import argparse
import sacrebleu
import numpy as np
from shapely.geometry import Polygon


def calculate_iou(box1, box2):
    """
    计算两个四边形的IoU值
    """
    # 将输入的四边形坐标转换为Polygon对象
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    # 计算交集区域的面积
    intersection_area = poly1.intersection(poly2).area

    # 计算两个四边形的面积
    area1 = poly1.area
    area2 = poly2.area

    # 计算并集区域的面积
    union_area = area1 + area2 - intersection_area

    # 计算IoU值
    iou = intersection_area / union_area

    # print("iou:", iou)
    return iou

def match_boxes(boxes1, boxes2, iou_threshold=0.5):
    """
    对两个图片中的文本框进行匹配
    """
    matches = []

    for i, box1 in enumerate(boxes1):
        max_iou = 0
        max_j = -1

        for j, box2 in enumerate(boxes2):
            iou = calculate_iou(box1, box2)

            if iou > max_iou:
                max_iou = iou
                max_j = j
        # print("max_iou:", max_iou)
        if max_iou > iou_threshold:
            matches.append((i, max_j))
        else:
            matches.append((i, -1))

    return matches

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_dir", type=str, default=None, help="generated img dir.")
    parser.add_argument("--ref_dir", type=str, default=None, help="reference img dir.")
    parser.add_argument("--lang", type=str, default='en', help="Source language.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold.")

    args = parser.parse_args()
    reader = easyocr.Reader([args.lang]) # this needs to run only once to load the model into memory

    # generate img read
    generate_result_folder = args.generate_dir
    
    generate_result_files = os.listdir(generate_result_folder)
    generate_result_files = [file for file in generate_result_files if 'jpg' in file]

    generate_result_files = sorted(generate_result_files, key=lambda x: int(x.split('.')[0]))
    generate_result_files = [os.path.join(generate_result_folder, file) for file in generate_result_files]
    # reference img read
    ref_result_dir = args.ref_dir
    
    ref_result_files = os.listdir(ref_result_dir)
    ref_result_files = [file for file in ref_result_files if 'jpg' in file]

    ref_result_files = sorted(ref_result_files, key=lambda x: int(x.split('.')[0]))
    ref_result_files = [os.path.join(ref_result_dir, file) for file in ref_result_files]
    
    # ocr
    generate_result = []
    ref_result = []
    for generate_file, ref_file in tqdm(zip(generate_result_files, ref_result_files)):
        generate_ocr_result = reader.readtext(generate_file, paragraph=True)
        ref_ocr_result = reader.readtext(ref_file, paragraph=True)
        generate_ocr_boxes = [item[0] for item in generate_ocr_result]
        ref_ocr_boxes = [item[0] for item in ref_ocr_result]

        matches = match_boxes(ref_ocr_boxes, generate_ocr_boxes, iou_threshold=args.iou_threshold)
        generate_ocr_result = [generate_ocr_result[item[1]][1] if item[1] != -1 else '' for item in matches]
        ref_ocr_result = [ref_ocr_result[item[0]][1] for item in matches]

        generate_text = ' '.join(generate_ocr_result)
        ref_text = ' '.join(ref_ocr_result)

        generate_result.append(generate_text)
        ref_result.append(ref_text)
    
    # calculate bleu
    bleu = sacrebleu.corpus_bleu(generate_result, [ref_result])
    print("iou threshold: {}".format(args.iou_threshold))
    print("structure sacrebleu: {}".format(bleu.score))