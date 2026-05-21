# 鸟类识别模型：
本项目基于inat2021鸟类子类数据，训练1486类鸟类识别模型，验证多个模型架构性能（见cfg文件夹），并导出onnx模型。
并基于Axera NPU 使用pulsar2进行**w8a16**量化, 导出axmodel模型。

## 模型训练

鸟类识别数据存放结构如：
```
.
├── train
│   ├── class_0001
│   │   ├── IMG_001.jpg
│   │   ├── IMG_002.jpg
│   │   ├── IMG_003.jpg
│   │   └── ...
│   ├── class_0002
│   │   ├── IMG_001.jpg
│   │   ├── IMG_002.jpg
│   │   ├── IMG_003.jpg
│   │   └── ...
│   ├── class_0003
│   │   └── ...
│   └── [1483 more directories]
└── val
    ├── class_0001
    │   ├── IMG_001.jpg
    │   ├── IMG_002.jpg
    │   ├── IMG_003.jpg
    │   └── ...
    ├── class_0002
    │   └── ...
    └── [1484 more directories]
```

通过cfg文件配置模型架构、超参数、数据集等，用户可根据需要选择不同的模型架构进行训练。
使用如下命令训练，代码支持torch、timm等框架模型。其中，timm模型的net_type与huggingface中释放的模型名称一致。
```
python train.py -c ./cfg/efficientnetv2_s.yaml
```
Benchmark:
![alt text](benchmark.png)

## 模型推理

验证识别模型的推理结果。
```
python demo.py -c ./cfg/efficientnetv2_s.yaml -m models/Birdmodel_inat_efficientnetv2_s_250311/model/final_model.pth --image test_images/04251_3a52191e-be71-4539-98ea-14a8f2347330.jpg
```

## 模型导出

导出onnx模型便于部署。
```
python export.py -c models/Birdmodel_inat_regnety_032_250318/config.yaml -m models/Birdmodel_inat_regnety_032_250318/model/best.pth
```

## onnx模型推理

onnx模型效果验证。
``` 
python onnxmodel/onnx_infer.py -m models/Birdmodel_inat_regnety_032_250318/regnety_032_finetuned.onnx -i test_images/04251_3a52191e-be71-4539-98ea-14a8f2347330.jpg
``` 
结果如下：
![alt text](prediction_result_top5.png)

## 模型量化

提供了鸟类检测+鸟类识别的量化脚本，其中鸟类检测使用的是基于yolov5的鸟类检测模型，在此不做赘述。
量化数据为训练/测试使用的鸟类图片，执行如下命令进行量化：
```
pulsar2 build --config ./axmodel/bird_det.json
pulsar2 build --config ./axmodel/bird_rec.json
```

## 板端部署
### axmodel单张/批量精度评估
```
python3 quant_model_eval.py
python3 quant_model_eval.py --image ./test_images/04251_3a52191e-be71-4539-98ea-14a8f2347330.jpg
```
运行结果如下：
```
root@ax630c:~/Bird# python3 quant_model_eval.py
[INFO] Available providers:  ['AxEngineExecutionProvider']
Ground truth loaded from ./val_list_flat.txt
build predictor with ./bird_rec.axmodel...
Loading ONNX model with providers: ['AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC20E
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.7.2a
[INFO] Model type: 0 (half core)
[INFO] Compiler version: 5.1-patch1 fa983fc0
Found 17893 images, starting inference (Top-5)...
100%|███████████████████████████████████████████████████████████████████████████████████| 17893/17893 [09:09<00:00, 32.57it/s]

=== 批量推理结果汇总 ===
总处理图片数: 17893
Top1正确数: 14365 | Top1准确率: 80.28%
Top5正确数: 16325 | Top5准确率: 91.24%
========================

```

### 检测+识别模型end2end推理

end2end demo由检测模型与识别模型组成:

| Models       | Platforms    | mAP@0.5       | latency      | CMM size(MB)  |
| -------------| -------------| --------------| -------------| --------------|
| bird_det     | AX630C       | 0.955         | 3.1ms        | 3.71          |

| Models       | Platforms    | latency      | Top1 Accuracy | Top5 Accuracy | CMM size(MB)  |
| -------------| -------------| -------------| --------------| --------------| --------------|
| bird_rec     | AX630C       | 16.3ms       | 81.3%           | 91.2%       | 31.8          |

使用以下脚本，进行推理。
```
python3 axmodel/axmodel_infer_end2end.py
```
运行结果如下：
```
(base) root@ax630c:~/Bird# python3 axmodel_infer_end2end.py
[INFO] Available providers:  ['AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC20E
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.7.2a
[INFO] Model type: 0 (half core)
[INFO] Compiler version: 5.1-patch1 fa983fc0
Loading ONNX model with providers: ['AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Model type: 0 (half core)
[INFO] Compiler version: 5.1-patch1 fa983fc0

========== Batch Processing Mode ==========
Input directory: ./test_images
Output directory: ./output
Found 7 images

[1/7] Processing: 03111_2c0dfa5a-c4a0-47f8-ac89-6a289208050f.jpg
  ✓ Detection: 1 bird(s)
    Bird #1: 03111_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Accipiter_badius (0.3655)
[2/7] Processing: 03332_01b365c3-a741-4f45-bac2-4345bc901ec6.jpg
  ✓ Detection: 1 bird(s)
    Bird #1: 04472_Animalia_Chordata_Aves_Piciformes_Ramphastidae_Ramphastos_toco (0.2589)
[3/7] Processing: 03412_0ffc115b-43b4-4474-a373-24233f391de3.jpg
  ✓ Detection: 1 bird(s)
    Bird #1: 03418_Animalia_Chordata_Aves_Charadriiformes_Laridae_Larus_glaucoides (0.7215)
[4/7] Processing: 03615_0dfbf6ae-434d-4648-b5d2-08412546ea64.jpg
  ✓ Detection: 1 bird(s)
    Bird #1: 03615_Animalia_Chordata_Aves_Galliformes_Cracidae_Ortalis_vetula (0.8448)
[5/7] Processing: 04251_3a52191e-be71-4539-98ea-14a8f2347330.jpg
  ✓ Detection: 1 bird(s)
    Bird #1: 04251_Animalia_Chordata_Aves_Passeriformes_Tityridae_Tityra_semifasciata (0.9079)
[6/7] Processing: 04593_3d74d5a7-15b1-4bb9-af6f-1bcd78485787.jpg
  ✓ Detection: 1 bird(s)
    Bird #1: 04593_Animalia_Chordata_Aves_Trogoniformes_Trogonidae_Trogon_elegans (0.9258)
[7/7] Processing: 3a52191e-be71-4539-98ea-14a8f2347330.jpg
  ✓ Detection: 1 bird(s)
    Bird #1: 04251_Animalia_Chordata_Aves_Passeriformes_Tityridae_Tityra_semifasciata (0.9079)

========== Summary ==========
Total images processed: 7
Total images with detections: 7
Total birds detected and classified: 7
Crops saved to: ./output/crops
Results saved to: ./output/results.json

```

需要注意的是，纯识别模型使用inat数据原图送入训练，而为检测+识别pipline训练的模型，使用的是inat原图鸟类目标框外扩一定比例后裁剪得到的小图数据；对检测+识别pipline，为避免训练/推理分布不匹配，尝试过以下识别模型的训练：
1.鸟类原图直接作为训练数据；
2.鸟类原图检测出鸟类目标后直接crop保存为训练数据；
3.鸟类原图检测出鸟类目标后目标框外扩一定比例后crop保存为训练数据；
其中，方案1训练得到的验证精度最高，方案2最低，方案3折中。
分析原因为：
①方案1精度最高，因为保留了完整上下文，模型可学习到全局、局部、环境特征，且不存在检测框不准的问题；
②方案2精度最低，因为丢失了生态上下文如环境、姿态信息，检测框容易切掉翅膀、头部、尾巴等关键特征且检测框可能偏移；
③方案3精度折中，因为保留了主体+部分上下文形成自然过渡，避免crop关键特征，去除了大面积无效背景干扰；
经多次实验，使用鸟类检测--->目标框外扩--->crop_resize--->鸟类识别的pipline推理得分较高。
