import os
import csv

def collect_mp4_files(input_dir, output_csv):

    mp4_files = []

    # 遍历文件夹及其子文件夹
    # pattern = re.compile(r'^(?!mask).*\.mp4$')
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("mp4"):
                print("123")
                # 获取文件的绝对路径
                full_path = os.path.join(root, file)
                mp4_files.append(full_path)

    # 将文件路径写入 CSV 

    if os.path.exists(output_csv):
        with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for file_path in mp4_files:
                writer.writerow([file_path])
    else:
        with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file_path"])
            for file_path in mp4_files:
                writer.writerow([file_path])

    print(f"已成功收集 {len(mp4_files)} 个 .wav 文件，并保存到 {output_csv}")

input_directory = "/mnt/shanhai-ai/shanhai-workspace/jyutong/dataset/raw/HDTF/vid"  # 替换为实际文件夹路径
output_csv_file = "/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/test/1217/HDTF_vid.csv"  # 替换为实际 CSV 文件路径
# 16338
os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

collect_mp4_files(input_directory, output_csv_file)
