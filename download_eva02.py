"""
EVA-02 预训练权重下载脚本
支持多种下载方式，自动处理网络问题
"""

import os
import urllib.request
from pathlib import Path

# 配置
MODEL_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
OUTPUT_FILE = "eva02_large_patch14_448.pth"

print("=" * 60)
print("EVA-02 预训练权重下载工具")
print("=" * 60)
print(f"模型: {MODEL_NAME}")
print(f"输出文件: {OUTPUT_FILE}")
print()

# 方法 1: 使用 HuggingFace 镜像
def download_from_mirror():
    """使用国内镜像下载"""
    mirror_urls = [
        "https://hf-mirror.com/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k/resolve/main/model.safetensors",
    ]

    for url in mirror_urls:
        try:
            print(f"正在从镜像下载: {url}")
            print("这可能需要几分钟，文件较大约 1.3GB...")
            urllib.request.urlretrieve(url, OUTPUT_FILE)
            print(f"✅ 下载成功: {OUTPUT_FILE}")

            # 检查文件大小
            size = os.path.getsize(OUTPUT_FILE)
            print(f"文件大小: {size / 1024 / 1024:.1f} MB")

            # 如果是 safetensors 格式，需要转换
            if OUTPUT_FILE.endswith('.safetensors'):
                print("检测到 safetensors 格式，需要转换...")
                convert_safetensors_to_pytorch(OUTPUT_FILE)
                # 删除原文件
                os.remove(OUTPUT_FILE)
                OUTPUT_FILE = "eva02_large_patch14_448.pth"

            return True
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            continue

    return False

# 方法 2: 直接从 HuggingFace 下载
def download_from_huggingface():
    """直接从 HuggingFace 下载"""
    url = "https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k/resolve/main/model.safetensors"

    try:
        print(f"正在从 HuggingFace 下载...")
        print("如果下载失败，请使用方法三（手动下载）")
        urllib.request.urlretrieve(url, OUTPUT_FILE)
        print(f"✅ 下载成功: {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

# 方法 3: 手动下载指引
def show_manual_download_guide():
    """显示手动下载指引"""
    print()
    print("=" * 60)
    print("手动下载指引")
    print("=" * 60)
    print()
    print("由于网络问题，请手动下载：")
    print()
    print("步骤 1：访问以下链接")
    print("-" * 60)
    print("https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    print("-" * 60)
    print()
    print("步骤 2：点击 'Files and versions' 标签")
    print("步骤 3：下载 'model.safetensors' 文件")
    print("步骤 4：将文件重命名为 'eva02_large_patch14_448.pth'")
    print("步骤 5：放到当前目录")
    print()
    print("或者使用镜像站：")
    print("-" * 60)
    print("https://hf-mirror.com/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k")
    print("-" * 60)
    print()

def convert_safetensors_to_pytorch(safetensors_path):
    """将 safetensors 格式转换为 PyTorch 格式"""
    try:
        from safetensors.torch import save_file, load_file

        # 加载 safetensors
        state_dict = load_file(safetensors_path)

        # 保存为 PyTorch 格式
        pytorch_path = safetensors_path.replace('.safetensors', '.pth')
        torch.save(state_dict, pytorch_path)

        print(f"✅ 转换成功: {pytorch_path}")
        return True
    except ImportError:
        print("⚠️  需要安装 safetensors 库: pip install safetensors")
        print("或者手动转换，请查看:")
        print("https://huggingface.co/docs/safetensors/index")
        return False
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

# 主函数
def main():
    # 检查是否已存在
    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️  文件已存在: {OUTPUT_FILE}")
        choice = input("是否重新下载？(y/n): ")
        if choice.lower() != 'y':
            print("已取消")
            return

    # 尝试不同下载方式
    print("尝试下载方式...")
    print()

    # 方法 1: 镜像
    if download_from_mirror():
        return

    # 方法 2: 直接下载
    if download_from_huggingface():
        return

    # 方法 3: 手动下载
    show_manual_download_guide()

if __name__ == '__main__':
    try:
        import torch
        main()
    except ImportError:
        print("❌ 请先安装 PyTorch: pip install torch")
        print()
        show_manual_download_guide()
