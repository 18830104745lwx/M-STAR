#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成带有实际地图背景的网格划分图（成都版）
用于模型图绘制素材
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import contextily as ctx
import os
import glob
from PIL import Image

# 设置matplotlib参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False

class ChengduGridMapGenerator:
    """生成成都网格划分地图"""
    
    def __init__(self, data_dir, output_path='chengdu_grid_map.png', basemap_path='chengdu.png'):
        self.data_dir = data_dir
        self.output_path = output_path
        self.basemap_path = basemap_path
        
        # 成都核心城区边界
        self.core_bounds = {
            'lon_min': 103.994821,
            'lon_max': 104.134852,
            'lat_min': 30.614351,
            'lat_max': 30.746338
        }
        
        # 网格参数
        self.grid_rows = 20
        self.grid_cols = 20
    
    def load_sample_data(self, sample_ratio=0.1):
        """加载少量样本数据用于可视化"""
        print("正在加载样本数据...")
        
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到CSV文件")
        
        # 只读取第一个文件以加快速度
        df = pd.read_csv(csv_files[0])
        
        # 成都数据的列名是中文
        lng_col = '上车位置经度'
        lat_col = '上车位置纬度'
        
        if lng_col not in df.columns:
            # 尝试英文列名
            lng_col = 'starting_lng'
            lat_col = 'starting_lat'
        
        # 数据清洗
        df = df.dropna(subset=[lng_col, lat_col])
        
        # 采样
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
        
        # 过滤到核心区域
        core_df = df[
            (df[lng_col] >= self.core_bounds['lon_min']) &
            (df[lng_col] <= self.core_bounds['lon_max']) &
            (df[lat_col] >= self.core_bounds['lat_min']) &
            (df[lat_col] <= self.core_bounds['lat_max'])
        ].copy()
        
        # 统一列名
        core_df['lng'] = core_df[lng_col]
        core_df['lat'] = core_df[lat_col]
        
        print(f"样本数据: {len(core_df):,} 条记录")
        return core_df
    
    def create_grid_edges(self):
        """创建网格边界"""
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        return lon_edges, lat_edges
    
    def generate_with_fallback(self, dpi=300, try_online=True):
        """生成网格图，支持在线地图和本地地图回退"""
        print("=" * 80)
        print("成都网格划分地图生成器")
        print("=" * 80)
        
        # 加载数据
        print("\n步骤1: 加载样本数据")
        sample_data = self.load_sample_data()
        
        print("\n步骤2: 创建网格")
        lon_edges, lat_edges = self.create_grid_edges()
        print(f"网格配置: {self.grid_rows}×{self.grid_cols} = {self.grid_rows*self.grid_cols}个网格")
        
        print("\n步骤3: 生成地图")
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # 设置坐标轴范围（必须在添加底图前设置）
        ax.set_xlim(self.core_bounds['lon_min'], self.core_bounds['lon_max'])
        ax.set_ylim(self.core_bounds['lat_min'], self.core_bounds['lat_max'])
        
        basemap_added = False
        
        # 首先尝试在线地图
        if try_online:
            print("尝试加载在线地图...")
            providers = [
                (ctx.providers.OpenStreetMap.Mapnik, "OpenStreetMap"),
                (ctx.providers.CartoDB.Positron, "CartoDB.Positron"),
                (ctx.providers.CartoDB.Voyager, "CartoDB.Voyager"),
            ]
            
            for provider, provider_name in providers:
                try:
                    print(f"  尝试 {provider_name}...")
                    # 使用默认zoom，让contextily自动计算
                    ctx.add_basemap(ax, crs='EPSG:4326', source=provider, alpha=0.8)
                    print(f"  ✓ 成功加载 {provider_name}")
                    basemap_added = True
                    break
                except Exception as e:
                    print(f"  ✗ {provider_name} 失败: {str(e)[:100]}")
                    continue
        
        # 如果在线地图失败，尝试本地地图
        if not basemap_added and os.path.exists(self.basemap_path):
            try:
                print("\n尝试加载本地地图...")
                basemap_img = Image.open(self.basemap_path)
                ax.imshow(basemap_img, extent=[
                    self.core_bounds['lon_min'], self.core_bounds['lon_max'],
                    self.core_bounds['lat_min'], self.core_bounds['lat_max']
                ], alpha=0.9, aspect='auto', zorder=1)
                print(f"✓ 已加载本地地图: {self.basemap_path}")
                basemap_added = True
            except Exception as e:
                print(f"✗ 本地地图加载失败: {e}")
        
        # 如果都失败，使用渐变背景
        if not basemap_added:
            print("\n使用默认背景")
            # 创建渐变背景
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            gradient = np.vstack((gradient, gradient))
            ax.imshow(gradient, extent=[self.core_bounds['lon_min'], self.core_bounds['lon_max'],
                                        self.core_bounds['lat_min'], self.core_bounds['lat_max']],
                      aspect='auto', cmap='Blues_r', alpha=0.1, zorder=0)
            ax.set_facecolor('#f8fbff')
        
        # 绘制数据点（红色）
        if len(sample_data) > 0:
            ax.scatter(sample_data['lng'], sample_data['lat'],
                      s=8, alpha=0.7, c='#DC143C', edgecolors='white', 
                      linewidth=0.3, rasterized=True, zorder=5, label='Taxi Trips')
            print(f"✓ 已绘制 {len(sample_data):,} 个数据点")
        
        # 绘制网格线（蓝色）
        for lon in lon_edges:
            ax.axvline(lon, color='#0066CC', linewidth=1.5, alpha=0.9, zorder=6)
        for lat in lat_edges:
            ax.axhline(lat, color='#0066CC', linewidth=1.5, alpha=0.9, zorder=6)
        print(f"✓ 已绘制 {self.grid_rows}×{self.grid_cols} 网格")
        
        # 移除标签
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        # 移除边框
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # 保存标准版本
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"\n✓ 标准版本已保存: {self.output_path}")
        
        # 高清版本
        output_path_hd = self.output_path.replace('.png', '_HD.png')
        plt.savefig(output_path_hd, dpi=600, bbox_inches='tight',
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"✓ 高清版本已保存: {output_path_hd}")
        
        plt.close()
        
        # 打印统计信息
        print("\n" + "=" * 80)
        print("📊 地图信息")
        print("=" * 80)
        print(f"区域: 成都市核心城区")
        lon_span = (self.core_bounds['lon_max'] - self.core_bounds['lon_min']) * 92.0  # 成都纬度约30度
        lat_span = (self.core_bounds['lat_max'] - self.core_bounds['lat_min']) * 111.0
        print(f"覆盖范围: {lon_span:.2f} km × {lat_span:.2f} km")
        print(f"网格配置: {self.grid_rows}×{self.grid_cols} = {self.grid_rows*self.grid_cols}个网格")
        print(f"每个网格: {lon_span/self.grid_cols:.2f} km × {lat_span/self.grid_rows:.2f} km")
        print(f"数据点数: {len(sample_data):,} 条（采样）")
        print("=" * 80)
        
        return self.output_path


def main():
    """主函数"""
    # 设置路径
    data_dir = '/root/lanyun-tmp/data_code/raw_data/2016年11月成都网约车滴滴订单数据'
    output_path = '/root/lanyun-tmp/data_code/chengdu_grid_map.png'
    basemap_path = '/root/lanyun-tmp/data_code/chengdu.png'
    
    # 创建生成器
    generator = ChengduGridMapGenerator(data_dir, output_path, basemap_path)
    
    try:
        # 生成网格地图 - 尝试在线地图
        generator.generate_with_fallback(dpi=300, try_online=True)  # 使用在线地图
        
        print("\n✅ 生成完成！")
        print("\n输出文件:")
        print(f"  📄 标准版本 (300 DPI): {output_path}")
        print(f"  📄 高清版本 (600 DPI): {output_path.replace('.png', '_HD.png')}")
        print("\n使用说明:")
        print("  • 图片无标题、无坐标轴标签，适合作为论文配图")
        print("  • 网格线为蓝色（#0066CC），数据点为红色（#DC143C）")
        print("  • 使用在线地图 (CartoDB Positron) 作为背景")
        print("  • 适合用于论文、演示文稿等学术用途")
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
