#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SBUS数据模拟器
用于测试sbus_realtime_plotter.py
生成模拟的16通道SBUS数据
"""

import time
import random
import math
import sys


def generate_sbus_data(counter, mode='random'):
    """
    生成16个通道的SBUS数据
    
    参数:
        counter: 计数器，用于生成周期性数据
        mode: 生成模式 ('random', 'sine', 'mixed')
    
    返回:
        包含16个通道值的列表
    """
    channels = []
    
    for i in range(16):
        if mode == 'random':
            # 随机模式：在900-2100之间随机变化
            value = random.randint(900, 2100)
            
        elif mode == 'sine':
            # 正弦波模式：每个通道使用不同相位的正弦波
            base = 1500
            amplitude = 400
            phase = (2 * math.pi * i) / 16  # 每个通道不同的相位
            freq = 0.1 * (i + 1)  # 不同频率
            value = int(base + amplitude * math.sin(counter * freq + phase))
            
        elif mode == 'mixed':
            # 混合模式：部分通道固定，部分变化
            if i < 4:
                # 前4个通道：正弦波
                base = 1500
                amplitude = 300
                value = int(base + amplitude * math.sin(counter * 0.1 + i))
            elif i < 8:
                # 中间4个通道：锯齿波
                value = int(1000 + (counter * 10 + i * 100) % 1000)
            elif i < 12:
                # 再4个通道：方波
                value = 1200 if (counter // 10 + i) % 2 == 0 else 1800
            else:
                # 最后4个通道：随机
                value = random.randint(1000, 2000)
        
        else:
            value = 1500  # 默认中间值
        
        channels.append(value)
    
    return channels


def print_sbus_data(channels, format_type='format1'):
    """
    以指定格式打印SBUS数据
    
    参数:
        channels: 16个通道的值
        format_type: 输出格式类型
    """
    if format_type == 'format1':
        # 格式1: "ch1: 1500, ch2: 1600, ..."
        output = ', '.join([f"ch{i+1}: {channels[i]}" for i in range(16)])
        
    elif format_type == 'format2':
        # 格式2: "1500 1600 1200 ..."
        output = ' '.join([str(ch) for ch in channels])
        
    elif format_type == 'format3':
        # 格式3: "[1500, 1600, 1200, ...]"
        output = '[' + ', '.join([str(ch) for ch in channels]) + ']'
        
    elif format_type == 'format4':
        # 格式4: "Channel 1: 1500, Channel 2: 1600, ..."
        output = ', '.join([f"Channel {i+1}: {channels[i]}" for i in range(16)])
    
    else:
        output = str(channels)
    
    print(output)
    sys.stdout.flush()


def main():
    """主函数"""
    print("SBUS Data Simulator")
    print("=" * 60)
    print("Usage: python sbus_data_simulator.py | python sbus_realtime_plotter.py")
    print("=" * 60)
    print()
    
    # 配置参数
    data_mode = 'mixed'  # 可选: 'random', 'sine', 'mixed'
    format_type = 'format1'  # 可选: 'format1', 'format2', 'format3', 'format4'
    update_rate = 0.05  # 更新频率（秒）
    
    counter = 0
    
    try:
        while True:
            # 生成数据
            channels = generate_sbus_data(counter, mode=data_mode)
            
            # 打印数据
            print_sbus_data(channels, format_type=format_type)
            
            # 等待
            time.sleep(update_rate)
            counter += 1
            
    except KeyboardInterrupt:
        print("\nSimulator stopped", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
