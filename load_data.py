import re
import pandas as pd
import os

# 定义文件路径
input_file = "result.txt"

# 检查文件是否存在
if not os.path.exists(input_file):
    print(f"文件 {input_file} 不存在，请检查路径！")
    exit()

# 从文件读取数据
with open(input_file, "r", encoding="utf-8") as file:
    data_text = file.read()

# 打印结果
# print("读取到的数据如下：")
# print(data_text)

# 提取数据的正则表达式
pattern = re.compile(
    r"(?P<operation>[\w.+\-]+)\s*\n"  
    r"mse:(?P<mse>[\d.]+), mae:(?P<mae>[\d.]+), rse:(?P<rse>[\d.]+), time:(?P<time>[\d.]+)"
)

# 使用正则表达式提取数据
matches = pattern.findall(data_text)

# 构建 DataFrame
columns = ["Operation", "MSE", "MAE", "RSE", "Time"]
data = [match for match in matches]
df = pd.DataFrame(data, columns=columns)

# 导出到 Excel 文件
output_file = "parsed_data_results.xlsx"
df.to_excel(output_file, index=False)

print(f"数据已成功导出到 {output_file}")
