#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBUS 16通道实时数据绘图工具
从终端输出读取SBUS数据并实时绘制曲线
"""

import sys
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import queue


class SBUSRealtimePlotter:
    """SBUS实时数据绘图器"""
    
    def __init__(self, max_points=200, window_cols=4, window_rows=4):
        """
        初始化绘图器
        
        参数:
            max_points: 每条曲线保存的最大数据点数
            window_cols: 子图列数
            window_rows: 子图行数
        """
        self.max_points = max_points
        self.num_channels = 16
        
        # 为每个通道创建数据队列
        self.data_queues = [deque(maxlen=max_points) for _ in range(self.num_channels)]
        self.time_data = deque(maxlen=max_points)
        self.time_counter = 0
        
        # 创建数据队列用于线程间通信
        self.input_queue = queue.Queue()
        
        # 创建图形和子图
        self.fig, self.axes = plt.subplots(window_rows, window_cols, figsize=(16, 12))
        self.fig.suptitle('SBUS 16 Channels Real-time Monitor', fontsize=16, fontweight='bold')
        self.axes = self.axes.flatten()
        
        # 为每个通道初始化绘图线
        self.lines = []
        for i in range(self.num_channels):
            ax = self.axes[i]
            ax.set_title(f'Channel {i+1}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 2100)  # SBUS通道值范围通常是172-1811
            line, = ax.plot([], [], 'b-', linewidth=1.5)
            self.lines.append(line)
        
        plt.tight_layout()
        
    def parse_sbus_line(self, line):
        """
        解析包含SBUS数据的行
        
        支持的格式:
        1. "ch1: 1500, ch2: 1600, ch3: 1200, ..."
        2. "1500 1600 1200 1300 ..."
        3. "[1500, 1600, 1200, 1300, ...]"
        4. "Channel 1: 1500, Channel 2: 1600, ..."
        
        参数:
            line: 输入的文本行
            
        返回:
            包含16个通道数据的列表，如果解析失败返回None
        """
        # 清理行
        line = line.strip()
        if not line:
            return None
        
        # 尝试不同的解析模式
        
        # 模式1: "ch1: 1500, ch2: 1600, ..."
        pattern1 = r'ch\d+\s*:\s*(\d+)'
        matches = re.findall(pattern1, line, re.IGNORECASE)
        if len(matches) >= self.num_channels:
            return [int(m) for m in matches[:self.num_channels]]
        
        # 模式2: "Channel 1: 1500, Channel 2: 1600, ..."
        pattern2 = r'channel\s*\d+\s*:\s*(\d+)'
        matches = re.findall(pattern2, line, re.IGNORECASE)
        if len(matches) >= self.num_channels:
            return [int(m) for m in matches[:self.num_channels]]
        
        # 模式3: 查找所有数字 (空格分隔或逗号分隔)
        pattern3 = r'\d+'
        matches = re.findall(pattern3, line)
        if len(matches) >= self.num_channels:
            # 过滤掉明显不是SBUS值的数字（例如太小或太大）
            values = [int(m) for m in matches if 0 <= int(m) <= 2200]
            if len(values) >= self.num_channels:
                return values[:self.num_channels]
        
        return None
    
    def read_input_thread(self):
        """
        在单独线程中读取标准输入
        """
        print("Starting to listen for terminal input...")
        print("Supported data formats:")
        print("  1. ch1: 1500, ch2: 1600, ch3: 1200, ...")
        print("  2. 1500 1600 1200 1300 ...")
        print("  3. [1500, 1600, 1200, 1300, ...]")
        print("Press Ctrl+C to exit\n")
        
        try:
            for line in sys.stdin:
                self.input_queue.put(line)
        except KeyboardInterrupt:
            pass
    
    def update_plot(self, frame):
        """
        更新绘图（animation回调函数）
        
        参数:
            frame: 动画帧号
        """
        # 处理输入队列中的所有数据
        while not self.input_queue.empty():
            try:
                line = self.input_queue.get_nowait()
                values = self.parse_sbus_line(line)
                
                if values:
                    # 更新时间数据
                    self.time_data.append(self.time_counter)
                    self.time_counter += 1
                    
                    # 更新每个通道的数据
                    for i, value in enumerate(values):
                        self.data_queues[i].append(value)
                    
                    # 打印确认（可选）
                    print(f"[{self.time_counter}] Received {len(values)} channels data", end='\r')
                    
            except queue.Empty:
                break
        
        # 更新每条曲线
        if len(self.time_data) > 0:
            time_list = list(self.time_data)
            for i in range(self.num_channels):
                data_list = list(self.data_queues[i])
                if len(data_list) > 0:
                    self.lines[i].set_data(time_list, data_list)
                    
                    # 动态调整x轴范围
                    ax = self.axes[i]
                    if len(time_list) > 1:
                        ax.set_xlim(time_list[0], time_list[-1])
                    
                    # 动态调整y轴范围（可选）
                    if len(data_list) > 10:
                        ymin = min(data_list) - 50
                        ymax = max(data_list) + 50
                        ax.set_ylim(ymin, ymax)
        
        return self.lines
    
    def run(self, interval=50):
        """
        启动实时绘图
        
        参数:
            interval: 更新间隔（毫秒）
        """
        # 启动输入读取线程
        input_thread = threading.Thread(target=self.read_input_thread, daemon=True)
        input_thread.start()
        
        # 创建动画
        ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=interval,
            blit=False,
            cache_frame_data=False
        )
        
        # 显示图形
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nProgram terminated")
            sys.exit(0)


def main():
    """主函数"""
    print("=" * 60)
    print("SBUS 16 Channels Real-time Plotter")
    print("=" * 60)
    
    # 创建绘图器
    plotter = SBUSRealtimePlotter(max_points=200, window_cols=4, window_rows=4)
    
    # 运行
    plotter.run(interval=50)


if __name__ == "__main__":
    main()
