# 模型权重放置目录

本目录用于放置课程"参考材料"中提供的 YOLO26n 装甲板四关键点模型权重 `best.pt`。

```
weights/
└── best.pt        # 由课程参考材料下载后放入（不要提交到 Git，已被 .gitignore 排除）
```

未放置权重时运行 `infer.py` 会直接提示找不到文件：

```
[错误] 找不到模型权重: weights/best.pt
        请按 README「模型权重准备」把 best.pt 放到该位置，或用 --weights 指定其他路径。
```

如权重放在其他位置，可通过参数指定：

```bash
python infer.py --weights D:/models/best.pt --source data/armor.mp4
```
