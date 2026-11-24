"""生成启动器图标：深色圆角底 + 等距立方体 + 速度线。"""
from PIL import Image, ImageDraw

SZ = 512
img = Image.new("RGBA", (SZ, SZ), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 圆角背景
d.rounded_rectangle([8, 8, SZ - 8, SZ - 8], radius=104, fill=(22, 25, 32, 255),
                    outline=(56, 68, 90, 255), width=6)

# 等距立方体
cx, cy, s = 268, 250, 112
hw = int(s * 0.866)
h = int(s * 1.5)
A = (cx, cy - s)          # 上顶点
B = (cx + hw, cy - s // 2)  # 右上
C = (cx, cy)              # 中心
D = (cx - hw, cy - s // 2)  # 左上
Bot = (cx, cy + h)        # 下顶点
BotL = (cx - hw, cy + h - s // 2)
BotR = (cx + hw, cy + h - s // 2)

d.polygon([A, B, C, D], fill=(126, 199, 255, 255))       # 顶面最亮
d.polygon([D, C, Bot, BotL], fill=(62, 140, 226, 255))   # 左侧
d.polygon([C, B, BotR, Bot], fill=(38, 100, 178, 255))   # 右侧

# 速度线
for i, (y, w) in enumerate([(212, 92), (244, 64), (276, 44)]):
    d.rounded_rectangle([44, y, 44 + w, y + 13], radius=7,
                        fill=(76, 154, 255, 200 - i * 50))

img.save("assets/llama.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
img.resize((256, 256), Image.LANCZOS).save("assets/llama-256.png")
print("ok")
