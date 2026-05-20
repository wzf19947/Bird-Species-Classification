import os
import argparse
import torch
import torch.onnx
import timm
import yaml
from pathlib import Path
from onnxsim import simplify

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_model_for_export(model_name, num_classes, pretrained=False):
    """
    构建模型并加载权重
    """
    print(f"🔄 正在构建模型: {model_name} ...")
    
    # 创建模型
    # 如果 pretrained=True，timm 会自动下载并加载官方权重
    # 如果 pretrained=False，我们后续手动加载
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.0,       # 导出时关闭 Dropout
        drop_path_rate=0.0,  # 导出时关闭 Stochastic Depth
        exportable=True      # 开启 timm 的导出优化模式
    )
    
    return model

def export_onnx(model, output_path, input_size, opset=17):
    """
    执行 ONNX 导出
    """
    model.eval()  # 切换到推理模式 (关键！)
    
    # 创建假输入 (Batch=1, Channels=3, H, W)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    print(f"🚀 开始导出 ONNX (Opset {opset})...")
    print(f"   输出路径：{output_path}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['classes'],
            verbose=False,
        )
        
        # 验证导出是否成功
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ 导出成功！文件大小：{file_size:.2f} MB")
            
            # 简单检查模型结构
            import onnx
            onnx_model = onnx.load(output_path)
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified model check failed."
            onnx.save(model_simp, output_path)
            print(f"Successfully simplified and saved to: {output_path}")
            
            # 打印输入输出信息
            print("\n📋 模型输入输出信息:")
            for inp in onnx_model.graph.input:
                print(f"   输入: {inp.name} -> {[d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]}")
            for out in onnx_model.graph.output:
                print(f"   输出: {out.name} -> {[d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]}")
                
        else:
            print("❌ 导出失败：文件未生成。")
            
    except Exception as e:
        print(f"❌ 导出过程中发生错误: {e}")
        print("\n💡 提示:")
        print("   1. 如果是 'GridSample' 或 'Slicing' 错误，尝试增加 --opset 版本 (如 18)。")
        print("   2. 如果是 ConvNeXtV2/SwinV2，确保 timm 和 onnx 都是最新版。")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Export Timm Model to ONNX")
    
    # 必要参数
    parser.add_argument('-c', '--config', type=str, required=True, help='训练时的 config.yaml 路径 (即使是 pretrained 模式也需要用于获取模型名称和输入尺寸)')
    
    # 互斥参数组：要么加载 checkpoint，要么使用 pretrained
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-m', '--checkpoint', type=str, help='微调后的模型权重路径 (.pth)。如果不填且未指定 --pretrained，则报错（除非逻辑调整，这里保持原逻辑：非 pretrained 模式必须填）')
    group.add_argument('--pretrained', action='store_true', help='直接加载 timm 官方预训练权重 (忽略 -m 参数)')
    
    # 可选参数
    parser.add_argument('--opset', type=int, default=17, help='ONNX Opset 版本 (ConvNeXtV2 建议 17+)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='加载权重的设备')
    parser.add_argument('--num_classes', type=int, default=None, help='覆盖配置中的类别数 (仅在 pretrained 模式下有效，因为 pretrained 权重通常是 1000 类)')
    
    args = parser.parse_args()
    
    # 逻辑检查：如果不是 pretrained 模式，必须提供 checkpoint
    if not args.pretrained and not args.checkpoint:
        parser.error("如果不使用 --pretrained 标志，则必须提供 -m/--checkpoint 参数。")

    # 1. 加载配置
    print(f"📖 加载配置: {args.config}")
    config = load_config(args.config)
    
    model_name = config.get('net_type')
    work_dir = config.get('work_dir', './')
    os.makedirs(f'models/{work_dir}', exist_ok=True)
    
    # 确定类别数
    if args.pretrained:
        # 如果是官方预训练，默认是 1000 类 (ImageNet)，除非用户强制指定
        num_classes = args.num_classes if args.num_classes else 1000
        print(f"⚠️  检测到 --pretrained 模式：将加载官方权重，类别数设为: {num_classes}")
    else:
        # 否则使用配置文件中的类别数 (你的任务通常是 1486)
        num_classes = 1486 
        # 也可以从 config 读取，如果 config 里有明确定义
        
    # 获取输入尺寸
    input_size_cfg = config.get('input_size', [224, 224])
    if isinstance(input_size_cfg, int):
        input_size_cfg = [input_size_cfg, input_size_cfg]
        
    print(f"🔍 模型信息: {model_name} | 类别数: {num_classes} | 输入尺寸: {input_size_cfg}")
    
    # 2. 构建模型
    model = build_model_for_export(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=args.pretrained # 关键：传递标志位
    )
    
    # 3. 加载权重 (仅当非 pretrained 模式时执行)
    if not args.pretrained:
        print(f"⬇️  加载微调权重: {args.checkpoint}")
        device = torch.device(args.device)
        
        # 处理权重字典
        # checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # 清理 key
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # 加载权重
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"   权重加载完成。缺失键: {len(msg.missing_keys)}, 多余键: {len(msg.unexpected_keys)}")
        
        model.to(device)
    else:
        print("✅ 已加载官方预训练权重，无需手动加载 checkpoint。")
        model.to(torch.device(args.device))
    
    # 4. 执行导出
    # 输出文件名区分一下，避免覆盖
    suffix = "_pretrained" if args.pretrained else "_finetuned"
    output_filename = f'{model_name}{suffix}.onnx'
    output_path = os.path.join(f'models/{work_dir}', output_filename)
    
    export_onnx(
        model=model,
        output_path=output_path,
        input_size=input_size_cfg,
        opset=args.opset
    )

if __name__ == '__main__':
    main()