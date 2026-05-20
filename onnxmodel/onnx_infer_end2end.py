import numpy as np
import cv2
import os
import argparse
from PIL import Image
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import traceback
# Ensure English fonts are used to avoid warnings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def xywh2xyxy(x):
    
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  
    y[..., 1] = x[..., 1] - x[..., 3] / 2  
    y[..., 2] = x[..., 0] + x[..., 2] / 2  
    y[..., 3] = x[..., 1] + x[..., 3] / 2  
    return y

def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def nms_boxes(boxes, scores):
 
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
 
    areas = w * h
    order = scores.argsort()[::-1]
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
 
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
 
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.45)[0]
 
        order = order[inds + 1]
    keep = np.array(keep)
    return keep
 
def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
 
    
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  
        prediction = prediction[0]  
 
    bs = prediction.shape[0]  
    nc = prediction.shape[2] - nm - 5  
    xc = prediction[..., 4] > conf_thres  
 
    
    max_wh = 7680  
    max_nms = 30000  
    redundant = True  
    multi_label &= nc > 1  
    merge = False  
 
    mi = 5 + nc  
    output = [np.zeros((0, 6 + nm))] * bs
 
    for xi, x in enumerate(prediction):  
        x = x[xc[xi]]  
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros(len(lb), nc + nm + 5)
            v[:, :4] = lb[:, 1:5]  
            v[:, 4] = 1.0  
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  
            x = np.concatenate((x, v), 0)
 
        
        if not x.shape[0]:
            continue
 
        x[:, 5:] *= x[:, 4:5]  
 
        
        box = xywh2xyxy(x[:, :4])  
        mask = x[:, mi:]  
 
        
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
 
        else:  
            conf = np.max(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            j = np.argmax(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(box.shape[:1][0]) > conf_thres]
 
        
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]
 
        
        n = x.shape[0]  
        if not n:  
            continue
        index = x[:, 4].argsort(axis=0)[:max_nms][::-1]
        x = x[index]
 
        
        c = x[:, 5:6] * (0 if agnostic else max_wh)  
        boxes, scores = x[:, :4] + c, x[:, 4]  
        i = nms_boxes(boxes, scores)
        i = i[:max_det]  
 
        if merge and (1 < n < 3E3):  
            iou = box_iou(boxes[i], boxes) > iou_thres  
            weights = iou * scores[None]  
            x[i, :4] = np.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  
            if redundant:
                i = i[iou.sum(1) > 1]  
 
        output[xi] = x[i]
 
    return output

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    if isinstance(gain, (list, tuple)):
        gain = gain[0]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [0, 2]] /= gain
    coords[:, [1, 3]] /= gain
    clip_coords(coords[0:4], img0_shape)
    #coords[:, 0:4] = coords[:, 0:4].round()

    return coords

def clip_coords(boxes, img_shape, step=2):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # x1 (索引 0, 2, 4...) -> 限制在 0 到 宽度(img_shape[1]) 之间
    boxes[:, 0::step] = np.clip(boxes[:, 0::step], 0, img_shape[1])
    # y1 (索引 1, 3, 5...) -> 限制在 0 到 高度(img_shape[0]) 之间
    boxes[:, 1::step] = np.clip(boxes[:, 1::step], 0, img_shape[0])


def plot_one_box(x, im, color=None, label=None, line_thickness=3, steps=2, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            # label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def expand_box(xyxy, im_shape, expand_ratio=0.1):
    """
    外扩检测框坐标
    
    Args:
        xyxy: 原始坐标 [x1, y1, x2, y2]
        im_shape: 图像形状 (height, width)
        expand_ratio: 外扩比例 (0.1 = 10%)
    
    Returns:
        外扩后的坐标 [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    
    # 计算扩展量
    expand_w = w * expand_ratio
    expand_h = h * expand_ratio
    
    # 应用扩展
    x1_new = round(max(0, x1 - expand_w))
    y1_new = round(max(0, y1 - expand_h))
    x2_new = round(min(im_shape[1], x2 + expand_w))
    y2_new = round(min(im_shape[0], y2 + expand_h))
    
    return [x1_new, y1_new, x2_new, y2_new]

class BirdDetector:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.classes=['bird']
        self.nc=len(self.classes)
        self.no = self.nc + 5
        self.na =3
        self.nl =3
        self.anchors=[[10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326]]
        self.stride=[8,16,32]
        self.anchors = np.array(self.anchors, dtype=np.float32).reshape(3, 3, 2)
        self.anchors = self.anchors / np.array(self.stride, dtype=np.float32).reshape(3, 1, 1)

    def preprocess_image(self, img, img_size=(480, 480)):
        img, _, _ = letterbox(img, img_size, auto=False, stride=32)
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img, 0)
        img /= 255.0
        return img

    def model_inference(self, input=None):
        output = self.model.run(None, {self.input_name: input})
        return output

    def _make_grid(self, anchors, stride, nx=20, ny=20, i=0):
        na = 3
        shape = 1, na, ny, nx, 2  
        y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x, indexing='ij')
        grid = np.broadcast_to(np.stack((xv, yv), 2),shape) - 0.5  
        anchor_grid = (np.array(anchors[i]) * np.array(stride[i])).reshape(1, na, 1, 1, 2)
        anchor_grid = np.broadcast_to(anchor_grid,shape)
        return grid, anchor_grid
    
    def postprocess(self, preds, img_shape, im0):
        res_img = im0.copy()
        z = []  # inference output
        for i,pred in enumerate(preds):         
            bs, _, ny, nx = pred.shape
            pred = pred.reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            grid, anchor_grid = self._make_grid(self.anchors, self.stride, nx, ny, i)

            pred = sigmoid(pred)
            
            xy, wh, conf = pred[...,:2],pred[...,2:4],pred[...,4:]
            
            xy = (xy * 2 + grid) * self.stride[i]  
            wh = (wh * 2) ** 2 * anchor_grid  
            y = np.concatenate((xy, wh, conf), 4)
            z.append(y.reshape(bs, self.na * nx * ny, self.no))

        preds=np.concatenate(z, 1)
        detections = []
        preds = non_max_suppression(preds, 0.3, 0.45)
        for i, det in enumerate(preds):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img_shape[2:], det[:, :4], im0.shape)
                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # print('det:',xyxy, conf, cls)
                    int_coords = [int(tensor.item()) for tensor in xyxy]
                    # print(int_coords)
                    detections.append(int_coords)
                    c = int(cls)  # integer class
                    label =  f'{self.classes[c]} {conf:.2f}'
                    plot_one_box(xyxy, res_img, label=label, color=colors(c, True), line_thickness=2,steps=3, orig_shape=im0.shape[:2])

        return detections, res_img


class BirdClassifier:
    """Bird classification classifier based on ONNX Runtime"""

    def __init__(self, class_name_file, model_file, mean, std, image_size=224):
        """
        Initialize the classifier.
        Defaults to CPUExecutionProvider.
        """
        self.rgb_mean = mean
        self.rgb_std = std
        self.image_size = image_size
        self.classes = self.load_classes(class_name_file)
        providers = ['CPUExecutionProvider']
        print(f"Loading ONNX model with providers: {providers}")
        
        try:
            self.session = ort.InferenceSession(model_file, providers=providers)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        self.transform = self.get_transform_params()
    
    def load_classes(self,class_name_file):
        with open(class_name_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    
    def get_transform_params(self):
        mean = np.array(self.rgb_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(self.rgb_std, dtype=np.float32).reshape(1, 3, 1, 1)
        return {'mean': mean, 'std': std}
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((int(self.image_size), int(self.image_size)), Image.BICUBIC)
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        mean = self.transform['mean']
        std = self.transform['std']
        img_array = (img_array - mean) / std
        
        return img_array.astype(np.float32)

    # [新增] 处理 numpy 数组输入
    def preprocess_image_array(self, image_array):
        """预处理 numpy 数组格式的图片 (BGR 格式，来自 cv2)"""
        # 转换 BGR 到 RGB
        image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # 转为 PIL Image
        image = Image.fromarray(image)
        # 缩放到指定大小
        image = image.resize((int(self.image_size), int(self.image_size)), Image.BICUBIC)
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        mean = self.transform['mean']
        std = self.transform['std']
        img_array = (img_array - mean) / std
        
        return img_array.astype(np.float32)
    
    def predict_image_topk(self, image_path, k=5):
        input_data = self.preprocess_image(image_path)
        outputs = self.session.run(None, {self.input_name: input_data})
        
        logits = outputs[0]
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        probs_0 = probabilities[0]
        top_k_indices = np.argsort(probs_0)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            class_name = self.classes[idx]
            conf = float(probs_0[idx])
            results.append((class_name, conf))
            
        return results

    # [新增] 处理数组输入的预测函数
    def predict_array_topk(self, image_array, k=5):
        """对 numpy 数组进行 Top-k 预测"""
        input_data = self.preprocess_image_array(image_array)
        outputs = self.session.run(None, {self.input_name: input_data})
        
        logits = outputs[0]
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        probs_0 = probabilities[0]
        top_k_indices = np.argsort(probs_0)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            class_name = self.classes[idx]
            conf = float(probs_0[idx])
            results.append((class_name, conf))
            
        return results
    
    def predict_batch_topk(self, image_dir, k=5):
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        files = sorted([f for f in os.listdir(image_dir) if any(f.lower().endswith(ext) for ext in image_extensions)])
        print(f"Found {len(files)} images, starting inference (Top-{k})...")
        
        for filename in files:
            image_path = os.path.join(image_dir, filename)
            try:
                top_k_results = self.predict_image_topk(image_path, k=k)
                results.append({
                    'filename': filename,
                    'path': image_path,
                    'top_k': top_k_results
                })
            except Exception as e:
                print(f"Error processing image {filename}: {str(e)}")
                
        return results
    
    def _wrap_text(self, text, max_chars=25):
        """
        Helper function to wrap or truncate long text to fit in table cells.
        Tries to break at underscores or hyphens first.
        """
        if len(text) <= max_chars:
            return text
        
        # Try to find a good breaking point (underscore or hyphen) near the limit
        break_points = [i for i, char in enumerate(text[:max_chars]) if char in ['_', '-']]
        
        if break_points:
            # Break at the last found separator within the limit
            split_idx = break_points[-1] + 1
            return text[:split_idx] + "\n" + text[split_idx:]
        
        # If no good break point, just force split in the middle
        mid = max_chars // 2
        return text[:mid] + "-\n" + text[mid:]

    def visualize_prediction_topk(self, image_path, top_k_results, save_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        ax1.imshow(image)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.axis('off')
        
        table_data = []
        table_data.append(["Rank", "Class Name", "Confidence"])
        
        processed_rows = []
        for i, (cls_name, conf) in enumerate(top_k_results):
            rank = f"#{i+1}"
            conf_str = f"{conf:.4f} ({conf*100:.2f}%)"
            
            # Process long class names
            wrapped_name = self._wrap_text(cls_name, max_chars=28) # Increased limit slightly but allow wrapping
            processed_rows.append([rank, wrapped_name, conf_str])
        
        # Combine header and rows
        full_table_data = [table_data[0]] + processed_rows
        
        # Create table with specific column widths
        # Col widths: Rank (10%), Name (60%), Conf (30%)
        table = ax2.table(cellText=full_table_data[1:], 
                          colLabels=full_table_data[0], 
                          loc='center', 
                          cellLoc='left', # Left align for text content usually looks better with wraps
                          colWidths=[0.1, 0.6, 0.3], 
                          bbox=[0.05, 0.1, 0.9, 0.75]) # Adjusted bbox to give more vertical space
        
        table.auto_set_font_size(False)
        
        # Dynamically adjust font size if names are very long/wrapped
        base_font_size = 10
        if any('\n' in row[1] for row in processed_rows):
            base_font_size = 8 # Reduce font if wrapping occurred
            
        table.set_fontsize(base_font_size)
        
        # Scale row height to accommodate wrapped text
        # Base scale 1.5, increase if wrapped
        row_scale = 1.8 if any('\n' in row[1] for row in processed_rows) else 1.5
        table.scale(1, row_scale)
        
        # Style the header
        for i in range(3):
            cell = table[(0, i)]
            cell.set_text_props(fontweight='bold', color='white', ha='center')
            cell.set_facecolor('#4472C4')
            if i == 1: # Center the header of the name column
                cell.set_text_props(ha='center')
            
        # Style body cells
        for i in range(1, len(full_table_data)):
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor('#ffffff' if i % 2 == 0 else '#f9f9f9')
                cell.set_edgecolor('#dddddd')
                cell.set_linewidth(1)
                
                # Alignment logic
                if j == 0: # Rank
                    cell.set_text_props(ha='center', va='center')
                elif j == 1: # Name (Left aligned, top aligned for wrapped text)
                    cell.set_text_props(ha='left', va='top', wrap=True)
                else: # Confidence
                    cell.set_text_props(ha='center', va='center')

        # Add File Path Text
        display_path = image_path
        if len(display_path) > 50:
            display_path = "..." + display_path[-47:]
            
        path_text = f"File Path:\n{display_path}"
        ax2.text(0.5, 0.92, path_text, 
                 ha='center', va='center', fontsize=9, color='#555555',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#eeeeee", ec="#cccccc", alpha=0.8))
        
        ax2.set_title('Top-5 Prediction Results', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        out_path = save_path if save_path else 'prediction_result_top5.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Result saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime Bird Detection + Crop + Classification (Batch Mode)")
    parser.add_argument("-c", "--class_map_file", 
                       default="./class_name.txt",
                       help="Path to bird class mapping file")
    parser.add_argument("--det_model", 
                       default="./bird_det.onnx", 
                       help="Path to detection model file")
    parser.add_argument("--rec_model", 
                       default="./bird_rec.onnx", 
                       help="Path to classification model file")
    parser.add_argument("--det_image_size", 
                       default=480, 
                       help="detection model input size")
    parser.add_argument("--rec_image_size", 
                       default=224, 
                       help="classification model input size")
    parser.add_argument("-mean", "--mean", 
                       type=float, 
                       nargs='+', 
                       default=[0.485, 0.456, 0.406], 
                       help="Mean normalization values of classification model")
    parser.add_argument("-std", "--std", 
                       type=float, 
                       nargs='+', 
                       default=[0.229, 0.224, 0.225], 
                       help="Standard deviation normalization values of classification model")
    parser.add_argument("--image_dir", 
                       default="../test_images",
                       help="Directory containing test images")
    parser.add_argument("-i", "--image", 
                       help="Path to a single test image")
    parser.add_argument("--top_k",
                       type=int,
                       default=5,
                       help="Number of top predictions to show (default: 5)")
    parser.add_argument("--expand_ratio",
                       type=float,
                       default=0.3,
                       help="Expansion ratio for bounding box (default: 0.3)")
    parser.add_argument("--output_dir",
                       default="./output",
                       help="Directory to save crops and results")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    crops_dir = os.path.join(args.output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    detector = BirdDetector(args.det_model)
    classifier = BirdClassifier(args.class_map_file, args.rec_model, args.mean, args.std, args.rec_image_size)
    
    if args.image and os.path.exists(args.image):
        try:
            print(f"\n========== Processing Single Image: {args.image} ==========")
            im0 = cv2.imread(args.image)
            if im0 is None:
                print(f"Error: Cannot read image {args.image}")
                return
            
            img = detector.preprocess_image(im0, img_size=(args.det_image_size, args.det_image_size))
            preds = detector.model_inference(img)
            det_result, res_img = detector.postprocess(preds, img.shape, im0)
            cv2.imwrite(os.path.join(args.output_dir,os.path.basename(args.image)), res_img)
            print(f"✓ Detection: Found {len(det_result)} bird(s)")
            
            if len(det_result) == 0:
                print("No birds detected in this image.")
                return
            
            det_result_expand = [expand_box(xyxy, im0.shape, args.expand_ratio) for xyxy in det_result]
            
            # 保存抠图和识别结果
            for bird_idx, box in enumerate(det_result_expand):
                x1, y1, x2, y2 = box
                cropped_image = im0[y1:y2, x1:x2]
                
                if cropped_image.size == 0:
                    print(f"  Bird #{bird_idx+1}: Invalid crop")
                    continue
                
                # 预测
                top_k_results = classifier.predict_array_topk(cropped_image, k=args.top_k)
                
                # 保存抠图
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                crop_save_path = os.path.join(crops_dir, f"{base_name}_bird_{bird_idx+1}.jpg")
                cv2.imwrite(crop_save_path, cropped_image)
                
                # 打印识别结果
                print(f"\n  Bird #{bird_idx+1}:")
                print(f"    Box: [{x1}, {y1}, {x2}, {y2}]")
                print(f"    Crop saved: {crop_save_path}")
                print(f"    Top-{args.top_k} Predictions:")
                for i, (cls_name, conf) in enumerate(top_k_results):
                    print(f"      #{i+1}: {cls_name} ({conf:.4f}, {conf*100:.2f}%)")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
    
    elif os.path.exists(args.image_dir):
        # 批处理模式
        print(f"\n========== Batch Processing Mode ==========")
        print(f"Input directory: {args.image_dir}")
        print(f"Output directory: {args.output_dir}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = sorted([f for f in os.listdir(args.image_dir) 
                       if any(f.lower().endswith(ext) for ext in image_extensions)])
        
        print(f"Found {len(files)} images\n")
        
        all_results = []
        total_birds = 0
        
        for file_idx, filename in enumerate(files, 1):
            image_path = os.path.join(args.image_dir, filename)
            print(f"[{file_idx}/{len(files)}] Processing: {filename}")
            
            try:
                im0 = cv2.imread(image_path)
                if im0 is None:
                    print(f"  ⚠ Skipped: Cannot read image")
                    continue
                
                img = detector.preprocess_image(im0, img_size=(args.det_image_size, args.det_image_size))
                preds = detector.model_inference(img)
                det_result, res_img = detector.postprocess(preds, img.shape, im0)
                cv2.imwrite(os.path.join(args.output_dir,filename), res_img)
                print(f"  ✓ Detection: {len(det_result)} bird(s)")
                
                if len(det_result) == 0:
                    continue
                
                det_result_expand = [expand_box(xyxy, im0.shape, args.expand_ratio) for xyxy in det_result]
                
                # 创建该图片的结果字典
                image_result = {
                    'filename': filename,
                    'image_path': image_path,
                    'num_birds': len(det_result),
                    'birds': []
                }
                
                # 处理每只检测到的鸟
                for bird_idx, box in enumerate(det_result_expand):
                    x1, y1, x2, y2 = box
                    cropped_image = im0[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        continue
                    
                    # 预测
                    top_k_results = classifier.predict_array_topk(cropped_image, k=args.top_k)
                    
                    # 保存抠图
                    base_name = os.path.splitext(filename)[0]
                    crop_save_path = os.path.join(crops_dir, f"{base_name}_bird_{bird_idx+1}.jpg")
                    cv2.imwrite(crop_save_path, cropped_image)
                    
                    # 记录结果
                    bird_result = {
                        'bird_id': bird_idx + 1,
                        'box': [x1, y1, x2, y2],
                        'crop_path': crop_save_path,
                        'predictions': top_k_results
                    }
                    image_result['birds'].append(bird_result)
                    total_birds += 1
                    
                    # 打印识别结果
                    print(f"    Bird #{bird_idx+1}: {top_k_results[0][0]} ({top_k_results[0][1]:.4f})")
                
                all_results.append(image_result)
                
            except Exception as e:
                print(f"  ⚠ Error: {e}")
                continue
        
        # 打印汇总结果
        print(f"\n========== Summary ==========")
        print(f"Total images processed: {len(files)}")
        print(f"Total images with detections: {len(all_results)}")
        print(f"Total birds detected and classified: {total_birds}")
        print(f"Crops saved to: {crops_dir}")
        
        # 保存详细结果到 JSON
        results_json_path = os.path.join(args.output_dir, "results.json")
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {results_json_path}")
    
    else:
        print("Error: Specified image or directory not found.")
        print(f"  Image: {args.image}")
        print(f"  Directory: {args.image_dir}")


if __name__ == "__main__":
    main()