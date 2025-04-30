import os


# def list_files_and_write_to_txt(directory, output_file):
#     """
#     列出指定文件夹中的所有文件名称，并将这些名称写入一个文本文件。
#
#     参数:
#         directory (str): 要读取的文件夹路径。
#         output_file (str): 输出的文本文件路径。
#     """
#     try:
#         # 获取文件夹中的所有文件和子文件夹名称
#         entries = os.listdir(directory)
#
#         # 打开输出文件
#         with open(output_file, 'w') as f:
#             for entry in entries:
#                 # 检查是否是文件
#                 full_path = os.path.join(directory, entry)
#                 if os.path.isfile(full_path):
#                     # 写入文件名称
#                     f.write(entry + '\n')
#
#         print(f"文件夹 '{directory}' 中的文件名称已写入到 '{output_file}'。")
#
#     except FileNotFoundError:
#         print(f"文件夹 '{directory}' 不存在。")
#     except PermissionError:
#         print(f"没有权限访问文件夹 '{directory}'。")


# 示例用法
directory_path = "path/to/your/folder"  # 替换为你的文件夹路径
output_file_path = "E:\PanoFormer-main\PanoFormer-main\PanoFormer\\depth.txt"  # 输出文件路径
root_path="E:\PanoFormer-main\PanoFormer-main\PanoFormer/data/panotodepth/train"
with open(output_file_path,"w")as f:
    for i in os.listdir(root_path):
        if "depth" in i :
            f.write(i+"\n")
def merge_files(file1, file2, output_file):
    """
    同时打开两个文件，并将每行内容合并到一起，写入输出文件。

    参数:
        file1 (str): 第一个文件的路径。
        file2 (str): 第二个文件的路径。
        output_file (str): 输出文件的路径。
    """
    try:
        # 打开两个输入文件和一个输出文件
        with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
            # 使用 zip 同时读取两个文件的行
            for line1, line2 in zip(f1, f2):
                # 合并两行内容，中间用空格或自定义分隔符连接
                merged_line = line1.strip() + " " + line2.strip() + "\n"
                # 写入输出文件
                out.write(merged_line)

        print(f"两个文件的内容已合并到 '{output_file}'。")

    except FileNotFoundError as e:
        print(f"文件未找到：{e}")
    except Exception as e:
        print(f"发生错误：{e}")


# 示例用法
file1_path = "E:\PanoFormer-main\PanoFormer-main\PanoFormer\\rgb.txt"
file2_path = "E:\PanoFormer-main\PanoFormer-main\PanoFormer\\depth.txt"
output_file_path = "rgb_depth_train.txt"

merge_files(file1_path, file2_path, output_file_path)
