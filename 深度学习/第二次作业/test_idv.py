import cv2
import numpy as np
import torch
from cv2 import WINDOW_NORMAL
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import Compose, Resize
from models.pspnet import PSPNet
from models.ccnet import CCNet
from models.deeplabv3 import DeepLabv3
import time
def mask_to_color(mask, colormap):
    """将类别掩膜转换为RGB颜色图像"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(colormap):
        color_mask[mask == i] = color
    return color_mask

def visualize_predictions(model, image_path, mask_path, colormap, class_names):
    # 加载并预处理图像
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image.astype(np.float32)/127.5 - 1).permute(2,0,1).unsqueeze(0).float()

    # 加载真实掩码
    true_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    true_mask = cv2.resize(true_mask, (128, 128), interpolation=cv2.INTER_NEAREST)

    # 推理
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        output = model(image_tensor)
        inference_time = time.perf_counter() - start

    # 获取预测结果
    pred_mask = output.argmax(1).squeeze().cpu().numpy()
    pred_color = mask_to_color(pred_mask, colormap)

    # 转换为BGR格式用于OpenCV显示
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    true_bgr = cv2.cvtColor(true_mask, cv2.COLOR_RGB2BGR)
    pred_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

    # 添加文字标注
    def add_text(img, text):
        return cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, (255, 255, 255), 1, cv2.LINE_AA)

    image_bgr = add_text(image_bgr, "Original Image")
    true_bgr = add_text(true_bgr, "Ground Truth")
    pred_bgr = add_text(pred_bgr, f"Predicted Mask ({inference_time:.4f}s)")

    # 水平拼接三张图
    combined = np.hstack([image_bgr, true_bgr, pred_bgr])

    # 显示并等待按键
    cv2.namedWindow("Segmentation Result",WINDOW_NORMAL)
    cv2.imshow("Segmentation Result", combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 配置参数
VOC_COLORMAP = [(0, 0, 0), (0, 0, 255), (0, 255, 0),
               (0, 128, 128), (255, 0, 0), (255, 0, 255)]
VOC_CLASSES = ['background', 'person', 'cat', 'plane', 'car', 'bird']

# 加载模型
model = PSPNet(6)
model.load_state_dict(torch.load('final_result/best_PSPNet_full.pth', map_location='cpu'))

# 运行可视化
visualize_predictions(
    model,
    './data/JPEGImages/00264.jpg',
    './data/Annotations/00264.png',
    colormap=VOC_COLORMAP,
    class_names=VOC_CLASSES
)