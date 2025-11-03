# SBUS 16通道实时数据绘图工具

## 简介

这是一个用于实时监控和可视化SBUS 16通道数据的Python工具。它可以从终端读取SBUS数据并以16个子图的形式实时绘制每个通道的数据曲线。

## 文件说明

- `sbus_realtime_plotter.py` - 主绘图程序，读取终端输入并实时绘制曲线
- `sbus_data_simulator.py` - 数据模拟器，用于测试和演示
- `README_SBUS.md` - 本文档

## 功能特点

✅ **实时绘图** - 支持实时更新16个通道的数据曲线  
✅ **多格式支持** - 自动识别多种SBUS数据格式  
✅ **动态缩放** - 自动调整坐标轴范围以适应数据  
✅ **高性能** - 使用线程和队列实现流畅的数据处理  
✅ **易于使用** - 通过管道直接从其他程序接收数据

## 安装依赖

```bash
# 安装必要的Python包
pip install matplotlib numpy

# 或者使用requirements.txt（如果有）
pip install -r requirements.txt
```

## 使用方法

### 方法1: 使用测试数据模拟器

```bash
# 运行模拟器并将输出传递给绘图工具
python sbus_data_simulator.py | python sbus_realtime_plotter.py
```

### 方法2: 从其他程序读取数据

```bash
# 从你的SBUS数据源程序读取数据
./your_sbus_program | python sbus_realtime_plotter.py
```

### 方法3: 手动输入（用于测试）

```bash
# 直接运行绘图工具
python sbus_realtime_plotter.py

# 然后手动输入数据，例如：
ch1: 1500, ch2: 1600, ch3: 1200, ch4: 1300, ch5: 1400, ch6: 1500, ch7: 1600, ch8: 1700, ch9: 1800, ch10: 1900, ch11: 1500, ch12: 1400, ch13: 1300, ch14: 1200, ch15: 1100, ch16: 1000
```

## 支持的数据格式

绘图工具可以自动识别以下格式：

1. **格式1** (推荐):
   ```
   ch1: 1500, ch2: 1600, ch3: 1200, ch4: 1300, ch5: 1400, ch6: 1500, ch7: 1600, ch8: 1700, ch9: 1800, ch10: 1900, ch11: 1500, ch12: 1400, ch13: 1300, ch14: 1200, ch15: 1100, ch16: 1000
   ```

2. **格式2** (空格分隔):
   ```
   1500 1600 1200 1300 1400 1500 1600 1700 1800 1900 1500 1400 1300 1200 1100 1000
   ```

3. **格式3** (数组格式):
   ```
   [1500, 1600, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 1500, 1400, 1300, 1200, 1100, 1000]
   ```

4. **格式4** (完整标签):
   ```
   Channel 1: 1500, Channel 2: 1600, Channel 3: 1200, ...
   ```

## 参数配置

### sbus_realtime_plotter.py

可以通过修改代码中的参数来自定义：

```python
plotter = SBUSRealtimePlotter(
    max_points=200,      # 每条曲线保存的最大数据点数
    window_cols=4,       # 子图列数
    window_rows=4        # 子图行数
)

plotter.run(interval=50)  # 更新间隔（毫秒）
```

### sbus_data_simulator.py

可以修改以下参数来改变模拟数据的特性：

```python
data_mode = 'mixed'        # 数据生成模式: 'random', 'sine', 'mixed'
format_type = 'format1'    # 输出格式: 'format1', 'format2', 'format3', 'format4'
update_rate = 0.05         # 数据更新频率（秒）
```

## 数据生成模式说明

- **random**: 所有通道随机生成900-2100之间的值
- **sine**: 每个通道使用不同频率和相位的正弦波
- **mixed**: 混合模式
  - 通道1-4: 正弦波
  - 通道5-8: 锯齿波
  - 通道9-12: 方波
  - 通道13-16: 随机值

## 示例应用场景

### 场景1: 监控无人机遥控器数据

```bash
# 假设你有一个读取SBUS数据的程序 sbus_reader
./sbus_reader /dev/ttyUSB0 | python sbus_realtime_plotter.py
```

### 场景2: 分析日志文件

```bash
# 从日志文件中提取SBUS数据并绘图
cat sbus_log.txt | python sbus_realtime_plotter.py
```

### 场景3: 网络数据流

```bash
# 从网络接收SBUS数据
nc -l 5000 | python sbus_realtime_plotter.py
```

## 高级用法

### 修改y轴范围

默认情况下，y轴会动态调整。如果要固定范围，可以修改代码：

```python
# 在 SBUSRealtimePlotter.__init__ 中
ax.set_ylim(0, 2100)  # 设置为SBUS标准范围 172-1811
```

### 更改显示的数据点数

```python
# 保留更多历史数据
plotter = SBUSRealtimePlotter(max_points=500)

# 或保留更少数据以提高性能
plotter = SBUSRealtimePlotter(max_points=100)
```

### 自定义子图布局

```python
# 单行显示所有通道
plotter = SBUSRealtimePlotter(window_cols=16, window_rows=1)

# 2x8布局
plotter = SBUSRealtimePlotter(window_cols=8, window_rows=2)
```

## 故障排除

### 问题1: 没有显示图形窗口

**解决方案**: 确保你的系统支持图形界面，如果是SSH连接，需要启用X11转发：
```bash
ssh -X user@host
```

### 问题2: 数据更新很慢

**解决方案**: 减少 `max_points` 或增加 `interval` 参数：
```python
plotter = SBUSRealtimePlotter(max_points=100)
plotter.run(interval=100)  # 增加更新间隔
```

### 问题3: 无法解析数据格式

**解决方案**: 检查你的数据格式是否包含至少16个数字。可以添加调试输出：
```python
# 在 parse_sbus_line 函数中添加
print(f"Debug: {line}")
```

## 性能建议

- 对于高频数据（>20Hz），建议设置 `max_points=100-200`
- 更新间隔 `interval` 建议设置为 50-100ms
- 如果CPU占用过高，可以增加更新间隔或减少保留的数据点

## 系统要求

- Python 3.6+
- matplotlib
- 图形界面支持（X11, Wayland等）

## 许可证

MIT License

## 贡献

欢迎提交问题和改进建议！

## 联系方式

如有问题，请提交Issue或Pull Request。
