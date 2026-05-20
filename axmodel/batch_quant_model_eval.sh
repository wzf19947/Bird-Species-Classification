# python3 quant_model_eval.py -m ./swint_rgb_224_u16.axmodel -imgsz 224
# python3 quant_model_eval.py -m ./convnext_tiny_rgb_384_u16.axmodel -imgsz 384
# python3 quant_model_eval.py -m ./convnext_small_rgb_384_u16.axmodel -imgsz 384
# python3 quant_model_eval.py -m ./vit_s_rgb_384_u16.axmodel -imgsz 384 --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5
# python3 quant_model_eval.py -m ./efficientnetb4_rgb_380_u16.axmodel -imgsz 380
# python3 quant_model_eval.py -m ./efficientnetv2s_rgb_384_u16.axmodel -imgsz 384
python3 quant_model_eval.py -m ./regy032_rgb_384_u16.axmodel -imgsz 384