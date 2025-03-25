import cv2
import time
import json
import os
import base64
from openai import OpenAI

# 初始化 OpenAI 客户端
# 分析图片
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_BUY"),
    base_url="https://api.chatanywhere.tech/v1"
)

# 求解移动步骤
client2 = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def capture_image(videoID):
    """使用摄像头拍照"""
    cap = cv2.VideoCapture(videoID) 
    if not cap.isOpened():
        raise Exception(f"无法打开摄像头/dev/video{videoID}")
    
    ret, frame = cap.read()
    if not ret:
        raise Exception(f"无法从/dev/video{videoID}获取图像")
    
    cap.release()
    return frame

def save_image(image, filename="hanoi_state.jpg"):
    """保存图像"""
    cv2.imwrite(filename, image)
    return filename

def encode_image(image_path):
    """将图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_step_1(image_path):
    # 第一步：请求分析图片
    base64_image = encode_image(image_path)
    prompt = """请分析这张汉诺塔游戏的图片。
                仅仅告诉我图片内汉诺塔当前的状态，
                包括柱子有多少个，以及各个柱子上圆盘的个数，相对大小顺序
                （一个柱子内圆盘大小顺序，和柱子之间圆盘大小顺序）
                请不要回答其他多余的文字，方便我进行后续的文字处理"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"第一步API请求失败: {str(e)}")


def analyze_image_step_2(analysis_result):
    # 第二步：在分析结果的基础上完成后续任务
    prompt = """基于分析结果，数字化表示汉诺塔圆盘的分布情况，
    格式如下：
    {"A": [], "B": [], "C": [], ...}
    注意：    
    1. 列表里面是圆盘的编号，顺序是升序
    2. 圆盘从1开始编号，编号越小的圆盘大小越大，编号不能重复。
    3. 不要回答其他多余的文字，方便我进行后续的文字处理。"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            # model="deepseek-chat",
            # model="gpt-4o",  

            messages=[
                {
                    "role": "user",
                    "content": prompt + f"\n分析结果: {analysis_result}"
                }
            ],
            max_tokens=8192
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"第二步API请求失败: {str(e)}")
    

def parse_states(content):
    """过滤文字"""
        
    # 提取JSON部分
    start = content.find('{')
    end = content.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError("未找到有效的JSON内容")
        
    json_str = content[start:end]
    states = json.loads(json_str)   
    return states

def send_to_arm(moves):
    """将移动顺序发送给机械臂"""
    print("发送给机械臂的移动顺序：")
    for move in moves:
        print(f"从{move['from']}移动到{move['to']}")
        # 实际实现中这里应该调用机械臂的控制接口
        time.sleep(1)  # 模拟机械臂移动时间

def get_state():
    try:
        # 1. 捕获图像
        print("正在捕获图像...")
        
        # 丢掉前几张，曝光错误的照片
        for i in range(4):
            image = capture_image(4)
        
        # 2. 保存图像
        image_path = save_image(image)
        print(f"图像已保存到：{image_path}")
        image_path="hanoi_state.jpg"
        
        # 3. 调用API分析
        analysis_result = analyze_image_step_1(image_path)
        print("图片状态识别：", analysis_result)

        api_response = analyze_image_step_2(analysis_result)
        
        # 4. 解析移动顺序
        states = parse_states(api_response)
        print("解析到状态：", states)
        
        # 5. 发送给机械臂
        # send_to_arm(moves)
        return states
        
    except Exception as e:
        print(f"发生错误：{str(e)}")



"""
# 分析结果示例

# o1-mini 数字化下面较为复杂的情况会错误 （但不作考虑，盘子初始状态只能全在某条柱子上）
analysis_result = "第一根柱子上有1个圆盘，颜色为绿色，大小最大。第二根柱子上有3个圆盘，从下到上颜色依次为：蓝色，青色，蓝色，大小依次减小。第三根柱子没有圆盘。"

#  较为简单的情况 基本求解正确（o1-mini、 deepseek-chat）
# analysis_result = "柱子共有三个。第一个柱子上有三个圆盘，从下到上依次为蓝色、粉色、浅蓝色，大小从大到小。第二个柱子和第三个柱子上没有圆盘。"

api_response = "第二步分析结果： {"A": [1, 2, 3], "B": [], "C": []}"
"""
