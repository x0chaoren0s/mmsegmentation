# https://mmsegmentation.readthedocs.io/zh_CN/latest/get_started.html#id4
# 不管效果，能跑通就算安装成功

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'checkpoints/benchmarks/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 从一个 config 配置文件和 checkpoint 文件里创建分割模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 测试一张样例图片并得到结果
img = 'datasets/2021-03-06-09-52-50/imgs/0000_h-m-s-ms=00-00-00-00.jpg'  # 或者 img = mmcv.imread(img), 这将只加载图像一次．
result = inference_segmentor(model, img)
# 在新的窗口里可视化结果
# model.show_result(img, result, show=True)
# 或者保存图片文件的可视化结果
# 您可以改变 segmentation map 的不透明度(opacity)，在(0, 1]之间。
model.show_result(img, result, out_file='output/demo_result.jpg', opacity=0.5)

# 测试一个视频并得到分割结果
# video = mmcv.VideoReader('../datasets/网箱清晰视频/1bc97b2af489af370ae805d3858145e7.mp4')
# for frame in video:
#    result = inference_segmentor(model, frame)
#    model.show_result(frame, result, wait_time=1, show=True)