#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch鸟类识别推理脚本
加载训练好的模型进行鸟类分类识别
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import timm
# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
from easydict import EasyDict as edict


class BirdPredictor:
    """鸟类识别预测器"""
    
    def __init__(self, config_file, model_file, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_file)
        self.classes = self.load_classes()
        self.model = self.load_model(model_file)
        self.transform = self.get_transform()
        
    def load_config(self, config_file):
        """加载配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return edict(config)
    
    def load_classes(self):
        """加载类别名称"""
        with open(self.config.class_name, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    
    def load_model(self, model_file):
        """加载模型"""
        import torch.nn as nn
        model = None
        net_type = self.config.net_type
        num_classes = len(self.classes)
        model_kwargs = {
            "num_classes": num_classes,
            "pretrained": False,
            "drop_rate": self.config.drop_rate,
            "drop_path_rate": self.config.drop_path_rate,
        }

        try:
            # 尝试使用 timm 加载
            print(f"正在通过 timm 加载模型: {net_type} ...")
            model = timm.create_model(net_type, **model_kwargs)
            if model is not None:
                model = model.to(self.device)
        except Exception as e:
            print(f"timm 加载 {net_type} 失败: {e}")
            print(f"使用原生 torchvision 或自定义模型...")

        
            if self.config.net_type == "resnet18":
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif self.config.net_type == "resnet34":
                model = models.resnet34(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif self.config.net_type == "resnet50":
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif self.config.net_type == "mobilenet_v2":
                model = models.mobilenet_v2(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif self.config.net_type == "googlenet":
                model = models.googlenet(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            else:
                raise ValueError(f"Unsupported model type: {self.config.net_type}")
        
        # 加载权重
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def get_transform(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize([self.config.input_size[1], self.config.input_size[0]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.rgb_mean, std=self.config.rgb_std),
        ])
    
    def predict_image(self, image_path):
        """预测单张图片"""
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            output_data = outputs.cpu().numpy()
            # with open("output0.bin", "wb") as f:
            #     f.write(output_data.tobytes())
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def predict_batch(self, image_dir):
        """批量预测目录中的图片"""
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_dir, filename)
                try:
                    pred_class, confidence = self.predict_image(image_path)
                    results.append({
                        'filename': filename,
                        'path': image_path,
                        'predicted_class': pred_class,
                        'confidence': confidence
                    })
                except Exception as e:
                    print(f"处理图片 {filename} 时出错: {str(e)}")
                    
        return results
    
    def visualize_prediction(self, image_path, predicted_class, confidence, save_path=None):
        """可视化预测结果"""
        # 加载图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示原图
        ax1.imshow(image)
        ax1.set_title('input')
        ax1.axis('off')
        
        # 显示预测结果
        ax2.text(0.5, 0.5, f'class: {predicted_class}\nscore: {confidence:.4f}', 
                ha='center', va='center', fontsize=16, transform=ax2.transAxes)
        ax2.set_title('result')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def create_demo_images(self):
        """创建演示用的示例图片"""
        demo_dir = "data/test_images"
        os.makedirs(demo_dir, exist_ok=True)
        
        # 创建一些示例图片用于演示
        print("创建演示图片...")
        
        # 创建简单的示例图片
        for i, bird_class in enumerate(self.classes[:5]):
            # 创建纯色背景图片
            img = np.random.randint(100, 255, (224, 224, 3), dtype=np.uint8)
            
            # 添加一些随机图案模拟鸟类
            cv2.circle(img, (112, 112), 50, (255, 255, 255), -1)
            cv2.ellipse(img, (112, 80), (30, 20), 0, 0, 360, (100, 100, 100), -1)
            
            # 保存图片
            filename = f"demo_{bird_class}_{i+1}.jpg"
            filepath = os.path.join(demo_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        return demo_dir


def main():
    parser = argparse.ArgumentParser(description="鸟类识别推理脚本")
    parser.add_argument("-c", "--config_file", 
                       default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("-m", "--model_file", 
                       default="./Birdmodel_inat_res18_250227/model/best_model_ep79_59.13189771197847_79.33378196500674.pth",
                       help="模型文件路径")
    parser.add_argument("--device", 
                       default="cuda:0", 
                       help="使用的设备")
    parser.add_argument("--image_dir", 
                       default="data/test_images",
                       help="测试图片目录")
    parser.add_argument('-i', "--image", 
                       help="单张测试图片路径")
    parser.add_argument("--create_demo", 
                       action="store_true",
                       help="创建演示图片")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config_file):
        print(f"配置文件不存在: {args.config_file}")
        return
        
    if not os.path.exists(args.model_file):
        print(f"模型文件不存在: {args.model_file}")
        print("请先生成训练好的模型文件")
        return
    
    # 创建预测器
    predictor = BirdPredictor(args.config_file, args.model_file, args.device)
    
    # 创建演示图片
    if args.create_demo:
        demo_dir = predictor.create_demo_images()
        print(f"已创建演示图片在: {demo_dir}")
    
    # 单张图片预测
    if args.image and os.path.exists(args.image):
        pred_class, confidence = predictor.predict_image(args.image)
        print(f"图片: {args.image}")
        print(f"预测类别: {pred_class}")
        print(f"置信度: {confidence:.4f}")
        
        # 可视化结果
        predictor.visualize_prediction(args.image, pred_class, confidence)
        print("预测结果已保存为: prediction_result.png")
    
    # 批量预测
    elif os.path.exists(args.image_dir):
        results = predictor.predict_batch(args.image_dir)
        
        print(f"\n共处理 {len(results)} 张图片:")
        print("-" * 60)
        for result in results:
            print(f"文件: {result['filename']}")
            print(f"预测: {result['predicted_class']} (置信度: {result['confidence']:.4f})")
            print("-" * 60)


if __name__ == "__main__":
    main()