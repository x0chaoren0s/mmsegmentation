
# # 生成默认完整config
# python train_copied_from_tools.py \
#     configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py

# python train_copied_from_tools.py \
#     configs/my_configs/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py
python test_copied_from_tools.py \
    configs/my_configs/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py \
    work_dirs/deeplabv3plus_r50-d8_512x512_40k_voc12aug/latest.pth \
    --work-dir work_dirs/deeplabv3plus_r50-d8_512x512_40k_voc12aug \
    --aug-test \
    --show-dir work_dirs/deeplabv3plus_r50-d8_512x512_40k_voc12aug/test2
