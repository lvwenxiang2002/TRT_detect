# TRT Inference Studio

多引擎 TensorRT 批量推理 GUI，基于 **Qt 6 + TensorRT + CUDA + OpenCV**。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 多任务面板 | 每个任务独立绑定一个 `.engine` 文件和一个图片文件夹 |
| 多线程推理 | 每个任务可独立配置线程数（1–16），共享同一 engine，各线程持有独立 `IExecutionContext` |
| 进度显示 | 每个任务拥有独立进度条，显示已处理 / 总图片数 |
| 全局控制 | "全部开始" / "全部停止" 一键操作所有任务 |
| 实时日志 | 右侧面板显示带时间戳的操作日志 |
| 结果输出 | 推理结果写入 `<folder>_results/` 目录（`.txt` 格式，可按需替换为 NMS 后处理逻辑） |

---

## 目录结构

```
trt_inference_gui/
├── CMakeLists.txt
└── src/
    ├── main.cpp
    ├── mainwindow.h / .cpp          ← 主窗口
    ├── enginetaskwidget.h / .cpp    ← 单任务 Widget（engine + 文件夹 + 进度条）
    ├── trtengine.h / .cpp           ← TensorRT engine 封装（线程安全）
    └── inferenceworker.h / .cpp     ← QRunnable 推理 worker
```

---

## 依赖

| 库 | 版本 | 说明 |
|----|------|------|
| Qt | 6.x | Core / Gui / Widgets / Concurrent |
| CUDA Toolkit | ≥ 11.8 | `CUDAToolkit` CMake 模块 |
| TensorRT | ≥ 8.x | `NvInfer.h`, `nvinfer.lib/.dll` |
| OpenCV | ≥ 4.5（可选）| 图像读取与预处理；不安装则回退到 stb_image |

---

## 构建步骤（Windows + MSVC）

```bat
:: 1. 克隆 / 解压源码
cd trt_inference_gui

:: 2. 设置 TensorRT 路径（修改为实际安装目录，例如 C:\TensorRT-10.x）
:: 在 CMakeLists.txt 中修改 TRT_ROOT，或在命令行传入：

cmake -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/msvc2022_64" ^
      -DTRT_ROOT="C:/TensorRT-10.x.x.x"

cmake --build build --config Release
```

### Linux（CUDA 版 Docker / 本机）

```bash
cmake -B build \
      -DCMAKE_PREFIX_PATH=/opt/Qt/6.x.x/gcc_64 \
      -DTRT_ROOT=/opt/TensorRT

cmake --build build -j$(nproc)
```

---

## 使用方法

1. 运行 `TRTInferenceGUI.exe`（或 Linux 可执行文件）。
2. 点击 **"＋ 添加任务"** 新增推理任务。
3. 每个任务：
   - 点击 **"浏览…"** 选择 `.engine` 文件 → 自动加载并显示输入尺寸。
   - 点击 **"浏览…"** 选择图片文件夹（支持 jpg / png / bmp / tiff）。
   - 调整 **线程数**（建议与 GPU SM 数量匹配，通常 4–8）。
4. 单击 **"▶ 开始推理"** 或 **"▶▶ 全部开始"**。
5. 推理结果保存在 `<图片文件夹>_results/` 中。

---

## 自定义后处理

在 `inferenceworker.cpp` 的 `saveResults()` 函数中替换占位实现，
接入 YOLO NMS / 标注保存 / 可视化等逻辑：

```cpp
void InferenceWorker::saveResults(const QString& imagePath,
                                  const std::vector<float>& output)
{
    // TODO: NMS → parse boxes → write YOLO txt / draw on image
}
```

---

## 注意事项

- Engine 文件须与运行机器的 GPU 架构和 TensorRT 版本匹配。
- 若 engine 使用动态 batch，请在 `TRTEngine::infer()` 中相应设置 binding shape。
- 多线程推理每个线程使用独立 `IExecutionContext`，GPU 显存占用会随线程数线性增长，请注意显存余量。
