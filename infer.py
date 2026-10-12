#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26n 装甲板四关键点检测 —— 视频推理与可视化脚本。

流程：
    1) 加载关键点（pose）权重，例如课程提供的 best.pt
       （类别 B1/B2/B3/R1/R2/R3，每个目标输出 4 个二维关键点）
    2) 读取自行拍摄的装甲板视频
    3) 逐帧推理，绘制检测框、类别、置信度，以及模型预测的 4 个关键点
    4) 保存与原视频同尺寸、同帧率、可正常播放的结果视频
    5) 可选：导出逐帧检测明细 CSV、保存代表性截图、输出统计信息

示例：
    python infer.py --weights weights/best.pt --source data/armor.mp4 --output outputs/armor_pred.mp4

全部参数见 README.md，或运行：python infer.py -h
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# 类别索引 -> 颜色（BGR），超出后循环使用
PALETTE = [
    (56, 56, 255),
    (0, 165, 255),
    (0, 200, 0),
    (255, 120, 0),
    (200, 0, 200),
    (0, 220, 220),
]

# 4 个关键点各自的颜色（按"关键点序号"着色，而不是按类别），便于区分四个角点
KPT_COLORS = [
    (0, 0, 255),    # 关键点 1：红
    (0, 255, 0),    # 关键点 2：绿
    (255, 0, 0),    # 关键点 3：蓝
    (0, 255, 255),  # 关键点 4：黄
]


# --------------------------------------------------------------------------- #
# 命令行参数
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26n 装甲板四关键点视频推理与可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("输入 / 输出")
    # --model 是 --weights 的别名，两种写法等价
    g.add_argument("--weights", "--model", "-w", default="weights/best.pt", help="模型权重路径")
    g.add_argument("--source", "-s", default="data/armor.mp4", help="输入视频路径")
    g.add_argument("--output", "-o", default="", help="结果视频路径（默认 outputs/<视频名>_pred.mp4）")
    g.add_argument("--codec", default="", help="视频编码器，默认依次尝试 mp4v / avc1 / XVID")
    g.add_argument("--out-scale", type=float, default=1.0,
                   help="结果视频缩放比例，1.0 = 与原视频同分辨率；例如 0.5 可把 4K 输出成 1080p 以减小文件体积")

    g = parser.add_argument_group("推理参数")
    g.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    g.add_argument("--iou", type=float, default=0.7, help="NMS 的 IoU 阈值")
    g.add_argument("--imgsz", type=int, default=640, help="推理输入分辨率")
    g.add_argument("--max-det", type=int, default=30, help="每帧最多保留的目标数")
    g.add_argument("--device", default="auto", help="auto / cpu / 0 / cuda:0")
    g.add_argument("--half", action="store_true", help="半精度推理（仅 GPU 有效）")
    g.add_argument("--classes", type=int, nargs="+", default=None,
                   help="只保留指定类别索引，例如 --classes 0 1 2")

    g = parser.add_argument_group("可视化")
    g.add_argument("--kpt-conf", type=float, default=0.0,
                   help="关键点置信度阈值，低于该值的关键点不绘制（0 = 全部绘制）")
    g.add_argument("--kpt-radius", type=int, default=0, help="关键点半径，0 = 按分辨率自动")
    g.add_argument("--line-width", type=int, default=0, help="线宽，0 = 按分辨率自动")
    g.add_argument("--no-connect", action="store_true", help="不绘制关键点之间的连线")
    g.add_argument("--no-kpt-index", action="store_true", help="不在关键点旁标注序号")

    g = parser.add_argument_group("额外输出")
    g.add_argument("--save-csv", action="store_true", help="导出逐帧检测明细 CSV")
    g.add_argument("--save-frames", type=int, default=5, help="保存的代表性截图数量，0 = 不保存")
    g.add_argument("--frame-dir", default="snapshots", help="截图保存目录")
    g.add_argument("--frame-gap", type=int, default=45, help="两张截图之间的最小帧间隔")
    g.add_argument("--stats-json", action="store_true", help="把统计信息写成 JSON 文件")
    g.add_argument("--keep-audio", action="store_true",
                   help="用 ffmpeg 把原视频音轨合并进结果视频（需要本机安装 ffmpeg）")
    g.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧，调试用")
    g.add_argument("--list-classes", action="store_true", help="打印模型类别后直接退出")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def fourcc(code: str) -> int:
    """兼容不同 OpenCV 版本的 fourcc 生成方式。"""
    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*code)
    return cv2.VideoWriter.fourcc(*code)


def open_writer(path: Path, codecs, fps: float, size):
    """依次尝试可用编码器创建 VideoWriter，返回 (writer, 实际编码器)。"""
    for code in codecs:
        writer = cv2.VideoWriter(str(path), fourcc(code), fps, size)
        if writer.isOpened():
            return writer, code
        writer.release()
    return None, None


def auto_style(width: int, height: int, args: argparse.Namespace) -> dict:
    """按视频分辨率推算线宽、关键点半径、字体大小，保证标记清晰可见。"""
    base = min(width, height)
    return {
        "lw": args.line_width if args.line_width > 0 else max(1, round(base / 480)),
        "radius": args.kpt_radius if args.kpt_radius > 0 else max(3, round(base / 160)),
        "fscale": max(0.4, base / 1100.0),
        "tth": max(1, round(base / 700.0 * 2)),
    }


def put_text(frame, text: str, org, fscale: float, thickness: int, color=(255, 255, 255)):
    """带黑色描边的文字，避免在浅色背景上看不清。"""
    cv2.putText(frame, text, org, FONT, fscale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, fscale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, dets, style: dict, args: argparse.Namespace) -> None:
    """把检测框、类别/置信度标签、模型预测的关键点画到帧上（原地修改）。"""
    lw = style["lw"]
    radius = style["radius"]
    fscale = style["fscale"]
    tth = style["tth"]

    for det in dets:
        cls_id = det["cls"]
        color = PALETTE[cls_id % len(PALETTE)]
        x1, y1, x2, y2 = (int(round(v)) for v in det["box"])

        # --- 1) 检测框 ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)

        # --- 2) 关键点：筛选"可见"的点 ---
        #     (0, 0) 是模型对缺失关键点的占位输出，直接跳过；
        #     其余按置信度阈值过滤。关键点坐标全部来自模型预测，不是检测框角点。
        visible = []
        kxy, kconf = det.get("kxy"), det.get("kconf")
        if kxy is not None:
            for i in range(len(kxy)):
                x, y = float(kxy[i][0]), float(kxy[i][1])
                c = float(kconf[i]) if (kconf is not None and i < len(kconf)) else 1.0
                if x <= 0.5 and y <= 0.5:
                    continue
                if c < args.kpt_conf:
                    continue
                visible.append((i, x, y, c))

        # --- 3) 关键点连线：4 个点连成闭合四边形，2~3 个点连成折线 ---
        #     线与框同色，便于看出关键点属于哪个目标。
        if len(visible) >= 2 and not args.no_connect:
            pts = np.array([[int(round(v[1])), int(round(v[2]))] for v in visible],
                           dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], len(visible) == 4, color, max(1, lw), cv2.LINE_AA)

        # --- 4) 关键点本体 + 序号 ---
        for i, x, y, _c in visible:
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(frame, (xi, yi), radius, KPT_COLORS[i % len(KPT_COLORS)], -1, cv2.LINE_AA)
            cv2.circle(frame, (xi, yi), radius, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
            if not args.no_kpt_index:
                put_text(frame, str(i + 1), (xi + radius + 2, yi - radius - 2),
                         fscale * 0.8, max(1, tth - 1))

        # --- 5) 类别 + 置信度标签 ---
        name = det["name"]
        label = f"{name} {det['conf']:.2f}"
        if not label.isascii():
            # cv2.putText 无法渲染中文类别名，退化为类别序号
            label = f"id{cls_id} {det['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(label, FONT, fscale, tth)
        ty = y1 - 6 if y1 - th - base - 6 >= 0 else y1 + th + base + 6
        cv2.rectangle(frame, (x1, ty - th - base), (x1 + tw + 8, ty + base), color, -1)
        put_text(frame, label, (x1 + 4, ty), fscale, tth)


def write_csv(path: Path, rows, nkpt: int) -> None:
    """把逐帧检测结果写成 CSV（便于在报告里做统计）。"""
    kpt_cols = []
    for i in range(nkpt):
        kpt_cols += [f"kp{i + 1}_x", f"kp{i + 1}_y", f"kp{i + 1}_conf"]
    header = ["frame", "time_sec", "cls_id", "class_name", "conf",
              "x1", "y1", "x2", "y2"] + kpt_cols

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            line = [row["frame"], f"{row['t']:.3f}", row["cls"], row["name"], f"{row['conf']:.4f}"]
            line += [f"{v:.1f}" for v in row["box"]]
            if row["kxy"] is None:
                line += [""] * (3 * nkpt)
            else:
                for i in range(nkpt):
                    if i < len(row["kxy"]):
                        x, y = row["kxy"][i]
                        conf = ""
                        if row["kconf"] is not None and i < len(row["kconf"]):
                            conf = f"{row['kconf'][i]:.3f}"
                        line += [f"{x:.1f}", f"{y:.1f}", conf]
                    else:
                        line += ["", "", ""]
            writer.writerow(line)


def mux_audio(video: Path, audio_source: Path, ffmpeg: str) -> bool:
    """用 ffmpeg 把原视频的音轨复制到结果视频里。"""
    tmp = video.with_name(video.stem + "_mux" + video.suffix)
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-i", str(video), "-i", str(audio_source),
           "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and tmp.exists():
        tmp.replace(video)
        return True
    if tmp.exists():
        tmp.unlink()
    return False


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    weights_path = Path(args.weights).expanduser()
    if not weights_path.is_file():
        print(f"[错误] 找不到模型权重: {weights_path}\n"
              f"        请按 README「模型权重准备」把 best.pt 放到该位置，或用 --weights 指定其他路径。",
              file=sys.stderr)
        return 2

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[错误] 缺少依赖: {exc}\n        请先执行: pip install -r requirements.txt",
              file=sys.stderr)
        return 2

    # ------------------------- 设备选择 -------------------------
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dev_desc = device
    if device.isdigit() and torch.cuda.is_available():
        dev_desc = f"{device} ({torch.cuda.get_device_name(int(device))})"
    print(f"[信息] torch {torch.__version__} | CUDA 可用: {torch.cuda.is_available()} | 使用设备: {dev_desc}")

    # ------------------------- 加载模型 -------------------------
    model = YOLO(str(weights_path))
    names = {int(k): str(v) for k, v in model.names.items()} if model.names else {}
    kpt_shape = getattr(getattr(model, "model", None), "kpt_shape", None)
    nkpt = int(kpt_shape[0]) if kpt_shape else 0

    print(f"[信息] 权重: {weights_path}")
    print(f"[信息] 任务类型: {getattr(model, 'task', '?')} | 关键点形状: {kpt_shape}")
    print(f"[信息] 类别数: {len(names)} -> {names}")
    if args.list_classes:
        return 0
    if nkpt and nkpt != 4:
        print(f"[警告] 该权重输出 {nkpt} 个关键点（本作业模型应为 4 个），请确认权重是否正确。")

    # ------------------------- 打开视频 -------------------------
    src = Path(args.source).expanduser()
    if not src.is_file():
        print(f"[错误] 找不到输入视频: {src}\n"
              f"        请按 README「视频拍摄说明」准备视频，或用 --source 指定其他路径。", file=sys.stderr)
        return 2
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {src}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps != fps or fps <= 1 or fps > 1000:
        print(f"[警告] 视频帧率异常（读取到 {fps}），结果视频按 30 FPS 写出。")
        fps = 30.0
    if width <= 0 or height <= 0:
        ok, probe = cap.read()
        if not ok:
            print("[错误] 视频无法读取任何帧。", file=sys.stderr)
            return 2
        height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"[信息] 视频: {src.name} | {width}x{height} | {fps:.2f} FPS | 约 {total} 帧")

    # ------------------------- 输出视频 -------------------------
    out_path = Path(args.output).expanduser() if args.output else Path("outputs") / f"{src.stem}_pred.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codecs = [args.codec] if args.codec else ["mp4v", "avc1", "XVID"]
    if args.out_scale <= 0:
        print(f"[警告] --out-scale 必须大于 0（当前为 {args.out_scale}），按 1.0 处理。")
        args.out_scale = 1.0
    out_size = (width, height)
    if abs(args.out_scale - 1.0) > 1e-6:
        # 宽高都取偶数，避免部分编码器对奇数尺寸报错
        out_size = (max(2, int(round(width * args.out_scale)) // 2 * 2),
                    max(2, int(round(height * args.out_scale)) // 2 * 2))
    writer, used_codec = open_writer(out_path, codecs, fps, out_size)
    if writer is None:
        print(f"[错误] 无法创建输出视频（已尝试编码器 {codecs}），可用 --codec 指定，例如 --codec mp4v",
              file=sys.stderr)
        return 2
    print(f"[信息] 输出视频: {out_path}（编码器 {used_codec}，{out_size[0]}x{out_size[1]}）")

    style = auto_style(width, height, args)
    # 推理参数只构造一次；half 仅在显式开启时传入（部分 ultralytics 版本已弃用该参数，
    # 默认不传可以避免逐帧打印弃用警告）
    predict_kwargs = dict(conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=device,
                          max_det=args.max_det, classes=args.classes, verbose=False, save=False)
    if args.half:
        predict_kwargs["half"] = True

    # ------------------------- 逐帧推理 -------------------------
    rows = []
    class_count = Counter()
    class_conf = defaultdict(float)
    kconf_sum, kconf_num = 0.0, 0
    frames_with_det = 0
    n_frames = 0
    speed = defaultdict(float)
    snaps = []
    last_snap = -(10 ** 9)
    warned_no_kpt = False

    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n_frames >= args.max_frames:
            break
        n_frames += 1

        result = model.predict(frame, **predict_kwargs)[0]

        dets = []
        boxes = result.boxes
        keypoints = getattr(result, "keypoints", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            kxy = keypoints.xy.cpu().numpy() if (keypoints is not None and keypoints.xy is not None) else None
            kc = None
            if keypoints is not None and getattr(keypoints, "conf", None) is not None:
                kc = keypoints.conf.cpu().numpy()
            if kxy is not None and nkpt == 0:
                nkpt = kxy.shape[1]
            for i in range(len(xyxy)):
                dets.append({
                    "box": xyxy[i],
                    "conf": float(confs[i]),
                    "cls": int(clss[i]),
                    "name": names.get(int(clss[i]), str(int(clss[i]))),
                    "kxy": kxy[i] if kxy is not None else None,
                    "kconf": kc[i] if kc is not None else None,
                })
        elif keypoints is None and not warned_no_kpt and n_frames == 1:
            warned_no_kpt = True
            print("[警告] 该权重没有输出关键点（可能不是 pose 模型），结果视频只会显示检测框。")

        if dets:
            frames_with_det += 1
            draw_detections(frame, dets, style, args)
            for det in dets:
                class_count[det["name"]] += 1
                class_conf[det["name"]] += det["conf"]
                if det["kconf"] is not None:
                    for c in det["kconf"]:
                        kconf_sum += float(c)
                        kconf_num += 1
                rows.append({
                    "frame": n_frames - 1,
                    "t": (n_frames - 1) / fps,
                    "cls": det["cls"],
                    "name": det["name"],
                    "conf": det["conf"],
                    "box": [float(v) for v in det["box"]],
                    "kxy": None if det["kxy"] is None else [[float(x), float(y)] for x, y in det["kxy"]],
                    "kconf": None if det["kconf"] is None else [float(c) for c in det["kconf"]],
                })
            if (args.save_frames > 0 and len(snaps) < args.save_frames
                    and (n_frames - 1) - last_snap >= args.frame_gap):
                snaps.append(n_frames - 1)
                last_snap = n_frames - 1
                Path(args.frame_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.frame_dir) / f"{src.stem}_frame{n_frames - 1:05d}.jpg"),
                            frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        # 需要缩小输出时，先在原分辨率上完成绘制再整体缩放，保证标记比例不变
        if out_size == (width, height):
            writer.write(frame)
        else:
            writer.write(cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA))
        for k, v in (result.speed or {}).items():
            speed[k] += float(v)
        if n_frames % 30 == 0 or (total and n_frames == total):
            print(f"\r[进度] {n_frames}/{total or '?'} 帧 | 含检测帧 {frames_with_det}", end="", flush=True)

    print()
    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t_start

    # ------------------------- 统计与附加输出 -------------------------
    if args.save_csv:
        csv_path = out_path.parent / f"{src.stem}_detections.csv"
        write_csv(csv_path, rows, nkpt)
        print(f"[信息] 检测明细 CSV: {csv_path}（{len(rows)} 条记录）")

    stats = {
        "weights": str(weights_path),
        "source": str(src),
        "output": str(out_path),
        "device": dev_desc,
        "frames": n_frames,
        "frames_with_detection": frames_with_det,
        "detections": int(sum(class_count.values())),
        "per_class": {k: {"count": v, "avg_conf": round(class_conf[k] / v, 4)}
                      for k, v in class_count.most_common()},
        "avg_keypoint_conf": round(kconf_sum / kconf_num, 4) if kconf_num else None,
        "avg_fps": round(n_frames / elapsed, 2) if elapsed > 0 else None,
        "conf_threshold": args.conf,
        "imgsz": args.imgsz,
    }
    if args.stats_json:
        json_path = out_path.parent / f"{src.stem}_stats.json"
        json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[信息] 统计信息 JSON: {json_path}")

    if args.keep_audio:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("[警告] 未找到 ffmpeg，跳过音轨合并（结果视频仅含画面）。")
        elif mux_audio(out_path, src, ffmpeg):
            print("[信息] 已将原视频音轨合并进结果视频。")
        else:
            print("[警告] 音轨合并失败，保留无声版本。")

    size_mb = out_path.stat().st_size / 1024 / 1024 if out_path.exists() else 0.0
    print("\n================ 推理结果汇总 ================")
    print(f"处理帧数          : {n_frames}")
    print(f"含检测的帧数      : {frames_with_det} ({frames_with_det / max(1, n_frames) * 100:.1f}%)")
    print(f"目标总数          : {sum(class_count.values())}")
    if class_count:
        print("分类别统计        :")
        for name, cnt in class_count.most_common():
            print(f"    {name:<8} {cnt:>4} 个    平均置信度 {class_conf[name] / cnt:.3f}")
    if kconf_num:
        print(f"关键点平均置信度  : {kconf_sum / kconf_num:.3f}")
    if elapsed > 0:
        print(f"平均处理帧率      : {n_frames / elapsed:.1f} FPS（含读帧、推理与绘图）")
    for key in ("preprocess", "inference", "postprocess"):
        if speed.get(key):
            print(f"平均 {key:<12}: {speed[key] / max(1, n_frames):.2f} ms/帧")
    print(f"结果视频          : {out_path}  ({size_mb:.1f} MiB)")
    if snaps:
        print(f"代表性截图        : {args.frame_dir}/ 共 {len(snaps)} 张")
    print("==============================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
