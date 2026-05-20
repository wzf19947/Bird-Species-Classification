#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axmodel Runtime Bird Classification Inference Script (Top-5 Enhanced)
Loads an exported axmodel model for bird classification.
Defaults to CPU execution.
"""
import os
import argparse
import numpy as np
import cv2
from PIL import Image
import axengine as axe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# Ensure English fonts are used to avoid warnings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False 

class BirdPredictor:
    """Bird classification predictor based on axmodel Runtime"""

    def __init__(self, class_name_file, model_file, mean, std, image_size=224):
        """
        Initialize the predictor.
        Defaults to AxEngineExecutionProvider.
        """
        self.rgb_mean = mean
        self.rgb_std = std
        self.image_size = image_size
        self.classes = self.load_classes(class_name_file)
        print(f"build predictor with {model_file}...")
        providers = ['AxEngineExecutionProvider']
        print(f"Loading axmodel model with providers: {providers}")
        
        try:
            self.session = axe.InferenceSession(model_file, providers=providers)
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

        img_array = np.array(image, dtype=np.uint8)
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
 
        return img_array
    
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
    
    def predict_batch_topk(self, image_dir, k=5):
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        files = sorted([f for f in os.listdir(image_dir) if any(f.lower().endswith(ext) for ext in image_extensions)])
        print(f"Found {len(files)} images, starting inference (Top-{k})...")
        
        for filename in tqdm(files):
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

    def calculate_batch_accuracy(self, batch_results, ground_truth_mapping=None):
        """
        计算批量图片的top1和top5准确率
        :param batch_results: predict_batch_topk返回的结果列表
        :param ground_truth_mapping: 可选，字典格式 {文件名: 真实类别名}
                                     如果未提供，默认从文件名中提取（假设文件名前缀是类别名，下划线/连字符分隔）
        :return: 包含top1/top5准确率的字典
        """
        if not batch_results:
            return {"top1_acc": 0.0, "top5_acc": 0.0, "total_images": 0, "correct_top1": 0, "correct_top5": 0}
        
        # 准备真实标签映射
        gt_mapping = {}
        if ground_truth_mapping is not None:
            gt_mapping = ground_truth_mapping
        else:
            # 自动从文件名提取真实标签（默认规则：取文件名第一个下划线/连字符前的部分）
            for res in batch_results:
                filename = res['filename']
                # 移除扩展名
                name_without_ext = os.path.splitext(filename)[0]
                # 按下划线/连字符分割取第一部分作为真实类别
                split_chars = ['_', '-', ' ']
                gt_class = name_without_ext
                for char in split_chars:
                    if char in gt_class:
                        gt_class = gt_class.split(char)[0]
                        break
                gt_mapping[filename] = gt_class.strip()
        
        total = len(batch_results)
        correct_top1 = 0
        correct_top5 = 0
        
        # 逐图验证top1/top5
        for res in batch_results:
            filename = res['filename']
            true_class = gt_mapping.get(filename, "")
            # print(f"----[GT]----:{filename}: {true_class}")
            top_k_preds = [cls_name for cls_name, _ in res['top_k']]
            # print(f"----[PRED]----:{filename}: {top_k_preds}")
            # 检查top1
            if true_class and true_class in top_k_preds[0]:
                correct_top1 += 1
            
            # 检查top5
            if true_class and true_class in top_k_preds[:5]:
                correct_top5 += 1
        
        # 计算准确率
        top1_acc = correct_top1 / total if total > 0 else 0.0
        top5_acc = correct_top5 / total if total > 0 else 0.0
        
        return {
            "total_images": total,
            "correct_top1": correct_top1,
            "correct_top5": correct_top5,
            "top1_acc": round(top1_acc, 4),
            "top5_acc": round(top5_acc, 4),
            "top1_acc_pct": f"{top1_acc*100:.2f}%",
            "top5_acc_pct": f"{top5_acc*100:.2f}%"
        }

def main():
    parser = argparse.ArgumentParser(description="axmodel Runtime Bird Classification (Top-5)")
    parser.add_argument("-c", "--class_map_file", 
                       default="./class_name.txt",
                       help="Path to configuration file")
    parser.add_argument("-m", "--model_file", 
                       default="./bird_rec.axmodel", 
                       help="Path to  model file")
    parser.add_argument("-imgsz", "--image_size", 
                       default=224, 
                       help="Input image size")
    parser.add_argument("-mean", "--mean", 
                       type=float, 
                       nargs='+', 
                       default=[0.485, 0.456, 0.406], 
                       help="Mean normalization values")
    parser.add_argument("-std", "--std", 
                       type=float, 
                       nargs='+', 
                       default=[0.229, 0.224, 0.225], 
                       help="Standard deviation normalization values")
    parser.add_argument("--image_dir", 
                       default="./valid",
                       help="Directory containing test images")
    parser.add_argument("--image", 
                       help="Path to a single test image")
    parser.add_argument("--top_k",
                       type=int,
                       default=5,
                       help="Number of top predictions to show (default: 5)")
    # 新增参数：指定真实标签文件（可选）
    parser.add_argument("--gt_file",
                       default='./val_list_flat.txt',
                       help="可选，真实标签文件路径，格式：每行 '文件名 真实类别名'")
    
    args = parser.parse_args()
    
    # 加载真实标签（如果提供）
    ground_truth = None
    if args.gt_file and os.path.exists(args.gt_file):
        ground_truth = {}
        with open(args.gt_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    ground_truth[parts[0]] = parts[1]
    print(f"Ground truth loaded from {args.gt_file}")
    # print("ground_truth:", ground_truth)
    predictor = BirdPredictor(args.class_map_file, args.model_file, args.mean, args.std, args.image_size)
    
    if args.image and os.path.exists(args.image):
        try:
            top_k_results = predictor.predict_image_topk(args.image, k=args.top_k)
            
            print(f"\nImage: {args.image}")
            print(f"Top-{args.top_k} Predictions:")
            for i, (cls_name, conf) in enumerate(top_k_results):
                print(f"#{i+1}: {cls_name} ({conf:.4f})")
            
            # predictor.visualize_prediction_topk(args.image, top_k_results)
            
        except Exception as e:
            print(f"Inference failed: {e}")
    
    elif os.path.exists(args.image_dir):
        results = predictor.predict_batch_topk(args.image_dir, k=args.top_k)
        # print('pred:', results)
        # 计算并打印批量top1/top5结果
        accuracy_stats = predictor.calculate_batch_accuracy(results, ground_truth)
        
        print(f"\n=== 批量推理结果汇总 ===")
        print(f"总处理图片数: {accuracy_stats['total_images']}")
        print(f"Top1正确数: {accuracy_stats['correct_top1']} | Top1准确率: {accuracy_stats['top1_acc_pct']}")
        print(f"Top5正确数: {accuracy_stats['correct_top5']} | Top5准确率: {accuracy_stats['top5_acc_pct']}")
        print(f"========================\n")
        
        # print(f"\nProcessed {len(results)} images:")
        # for res in results:
        #     print(f"File: {res['filename']}")
        #     for i, (cls_name, conf) in enumerate(res['top_k']):
        #         marker = "[1]" if i == 0 else "   "
        #         print(f"{marker} #{i+1}: {cls_name} ({conf:.4f})")
            
        # print("\nNote: Visualization saves only the last processed image in batch mode.")
        # if results:
        #     last_res = results[-1]
        #     predictor.visualize_prediction_topk(last_res['path'], last_res['top_k'], save_path='batch_last_result.png')

    else:
        print("Specified image or directory not found.")

if __name__ == "__main__":
    main()