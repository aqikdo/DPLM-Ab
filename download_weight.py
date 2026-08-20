import os
import requests
from tqdm import tqdm

# 创建目标目录
os.makedirs("checkpoints/dplm2_650m", exist_ok=True)

# 绕过所有 hub 库，直接通过镜像站的浏览器直链拉取最大的那个大文件
url = "https://hf-mirror.com/airkingbd/dplm2_650m/resolve/main/pytorch_model.bin"
output_path = "checkpoints/dplm2_650m/pytorch_model.bin"

print("正在绕过 SDK 直接下载权重文件...")
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(output_path, "wb") as file, tqdm(
    total=total_size, unit='iB', unit_scale=True, desc="下载进度"
) as bar:
    for data in response.iter_content(chunk_size=1024*1024):
        size = file.write(data)
        bar.update(size)

print("\n大文件下载完成！")