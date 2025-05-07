from PIL import Image  # 确保导入了 ImageResampling
import os

def resize_images_in_directory(directory, output_size=(100, 100)):
    """
    遍历指定目录下的所有图片文件，并将其大小调整为output_size指定的最大宽度和高度。
    
    :param directory: 包含图片的目录路径
    :param output_size: 输出图片的目标最大尺寸，默认是(100, 100)
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff')):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    # 计算新的尺寸，保持原始宽高比
                    ratio = min(output_size[0]/img.width, output_size[1]/img.height)
                    new_size = (int(img.width*ratio), int(img.height*ratio))
                    
                    # 调整图片大小
                    img_resized = img.resize(new_size, Image.LANCZOS)  # 使用 ImageResampling.LANCZOS 替代 ANTIALIAS
                    
                    # 保存调整后的图片，覆盖原图或另存为新文件
                    img_resized.save(file_path)
                    print(f"已调整大小: {filename}")
            except IOError:
                print(f"无法处理图片: {filename}")

# 使用方法
resize_images_in_directory("/home/jovyan/2024-srtp/srtp-final/logos/")