import cv2
import numpy as np
from PIL import Image

# JPEG壓縮攻擊
# input_path: 輸入圖片路徑
# output_path: 輸出圖片路徑
# quality: JPEG品質(1-100)，數字越小壓縮越嚴重
def jpeg_compress(input_path, output_path, quality=10):
    # 使用PIL開啟圖片並以指定品質存成JPEG
    img = Image.open(input_path)
    img.save(output_path, 'JPEG', quality=quality)
    print(f'JPEG compressed image saved to {output_path}')

# 縮放攻擊（未用於主流程，可擴充）
def resize_attack(input_path, output_path, scale=0.5):
    # 先縮小再放大回原尺寸，模擬壓縮失真
    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, img_resized)
    print(f'Resized image saved to {output_path}')

# 高斯雜訊攻擊
# input_path: 輸入圖片路徑
# output_path: 輸出圖片路徑
# noise_std: 標準差，越大雜訊越明顯
def noise_attack(input_path, output_path, noise_std=10):
    # 讀取灰階圖片並加入高斯雜訊
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    noise = np.random.normal(0, noise_std, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy_img)
    print(f'Noisy image saved to {output_path}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python attack_simulation.py attack_type input.png output.png')
        print('attack_type: jpeg | resize | noise')
    else:
        attack_type = sys.argv[1]
        if attack_type == 'jpeg':
            jpeg_compress(sys.argv[2], sys.argv[3])
        elif attack_type == 'resize':
            resize_attack(sys.argv[2], sys.argv[3])
        elif attack_type == 'noise':
            noise_attack(sys.argv[2], sys.argv[3])
        else:
            print('Unknown attack type!') 