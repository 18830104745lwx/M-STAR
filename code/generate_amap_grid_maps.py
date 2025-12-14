#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成带高德地图背景的网格地图（成都和海口）
使用英文版高德地图瓦片服务
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib import rcParams
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False

class AMapGridGenerator:
    """生成带高德地图背景的网格地图"""
    
    def __init__(self):
        # 成都核心城区边界
        self.chengdu_bounds = {
            'lon_min': 103.994821,
            'lon_max': 104.134852,
            'lat_min': 30.614351,
            'lat_max': 30.746338
        }
        
        # 海口核心城区边界
        self.haikou_bounds = {
            'lon_min': 110.277127,
            'lon_max': 110.376339,
            'lat_min': 19.978258,
            'lat_max': 20.060804
        }
        
        # 网格参数
        self.grid_rows = 20
        self.grid_cols = 20
        
        # 高德地图瓦片服务URL
        # style=7 矢量地图, style=8 路网地图, style=6 卫星图
        self.amap_urls = [
            # 高德矢量地图（标准版）
            'http://webrd01.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            'http://webrd02.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            'http://webrd03.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            'http://webrd04.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            # 备用: 高德路网地图
            'http://webrd01.is.autonavi.com/appmaptile?size=1&scale=1&style=8&x={x}&y={y}&z={z}',
            # 备用: CartoDB 淡色地图
            'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
            # 备用: OpenStreetMap
            'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        ]
    
    def load_chengdu_data(self, sample_ratio=0.1):
        """加载成都数据"""
        data_dir = '/root/lanyun-tmp/data_code/raw_data/2016年11月成都网约车滴滴订单数据'
        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        if not csv_files:
            return pd.DataFrame()
        
        df = pd.read_csv(csv_files[0])
        lng_col = '上车位置经度'
        lat_col = '上车位置纬度'
        if lng_col not in df.columns:
            lng_col = 'starting_lng'
            lat_col = 'starting_lat'
        
        df = df.dropna(subset=[lng_col, lat_col])
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
        
        core_df = df[
            (df[lng_col] >= self.chengdu_bounds['lon_min']) &
            (df[lng_col] <= self.chengdu_bounds['lon_max']) &
            (df[lat_col] >= self.chengdu_bounds['lat_min']) &
            (df[lat_col] <= self.chengdu_bounds['lat_max'])
        ].copy()
        
        core_df['lng'] = core_df[lng_col]
        core_df['lat'] = core_df[lat_col]
        return core_df
    
    def load_haikou_data(self, sample_ratio=0.1):
        """加载海口数据"""
        data_dir = '/root/lanyun-tmp/data_code/raw_data/海口打车数据'
        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        if not csv_files:
            return pd.DataFrame()
        
        df = pd.read_csv(csv_files[0], dtype={'order_id': str})
        required_cols = ['starting_wgs84_lng', 'starting_wgs84_lat']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        df = df.dropna(subset=required_cols)
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
        
        core_df = df[
            (df['starting_wgs84_lng'] >= self.haikou_bounds['lon_min']) &
            (df['starting_wgs84_lng'] <= self.haikou_bounds['lon_max']) &
            (df['starting_wgs84_lat'] >= self.haikou_bounds['lat_min']) &
            (df['starting_wgs84_lat'] <= self.haikou_bounds['lat_max'])
        ]
        return core_df
    
    def add_amap_basemap(self, ax, bounds):
        """添加高德地图底图"""
        # 设置坐标轴范围（必须在添加底图前设置）
        ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
        ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
        
        basemap_added = False
        
        print("  尝试加载高德地图...")
        
        # 尝试不同的地图源
        map_names = [
            "高德矢量地图 1",
            "高德矢量地图 2", 
            "高德矢量地图 3",
            "高德矢量地图 4",
            "高德路网地图",
            "CartoDB 淡色地图",
            "OpenStreetMap"
        ]
        
        for i, url in enumerate(self.amap_urls):
            try:
                if i < len(map_names):
                    print(f"    尝试 {map_names[i]}...")
                
                # 使用contextily添加自定义瓦片地图
                ctx.add_basemap(
                    ax,
                    crs='EPSG:4326',
                    source=url,
                    zoom=12,  # 适中的zoom级别
                    alpha=0.9,
                    attribution=""
                )
                
                print(f"    ✓ 成功加载 {map_names[i] if i < len(map_names) else '地图'}")
                basemap_added = True
                break
                
            except Exception as e:
                continue
        
        # 如果所有在线地图都失败，使用fallback背景
        if not basemap_added:
            print("  警告: 所有地图源均失败，使用默认背景")
            # 创建简单的灰色背景
            ax.set_facecolor('#f5f5f5')
            
            # 添加简单的网格作为背景
            lon_range = bounds['lon_max'] - bounds['lon_min']
            lat_range = bounds['lat_max'] - bounds['lat_min']
            
            # 背景细网格
            for i in range(50):
                y = bounds['lat_min'] + i * lat_range / 50
                ax.plot([bounds['lon_min'], bounds['lon_max']], [y, y], 
                       color='#e0e0e0', linewidth=0.3, alpha=0.5, zorder=0)
            for i in range(50):
                x = bounds['lon_min'] + i * lon_range / 50
                ax.plot([x, x], [bounds['lat_min'], bounds['lat_max']], 
                       color='#e0e0e0', linewidth=0.3, alpha=0.5, zorder=0)
        
        return basemap_added
    
    def generate_city_map(self, city_name, bounds, data, output_path, dpi=300):
        """生成单个城市的地图"""
        print(f"\n生成{city_name}地图...")
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # 添加高德地图背景
        self.add_amap_basemap(ax, bounds)
        
        # 绘制数据点（红色）
        if len(data) > 0:
            if city_name == '成都':
                ax.scatter(data['lng'], data['lat'],
                          s=8, alpha=0.7, c='#DC143C', edgecolors='white', 
                          linewidth=0.3, rasterized=True, zorder=5)
            else:  # 海口
                ax.scatter(data['starting_wgs84_lng'], data['starting_wgs84_lat'],
                          s=8, alpha=0.7, c='#DC143C', edgecolors='white', 
                          linewidth=0.3, rasterized=True, zorder=5)
            print(f"  ✓ 已绘制 {len(data):,} 个数据点")
        
        # 创建网格边界
        lon_edges = np.linspace(bounds['lon_min'], bounds['lon_max'], self.grid_cols + 1)
        lat_edges = np.linspace(bounds['lat_min'], bounds['lat_max'], self.grid_rows + 1)
        
        # 绘制网格线（蓝色，加粗）
        for lon in lon_edges:
            ax.axvline(lon, color='#0066CC', linewidth=2, alpha=0.9, zorder=6)
        for lat in lat_edges:
            ax.axhline(lat, color='#0066CC', linewidth=2, alpha=0.9, zorder=6)
        print(f"  ✓ 已绘制 {self.grid_rows}×{self.grid_cols} 网格")
        
        # 移除标签和边框
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # 保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"  ✓ 已保存: {output_path}")
        
        # 高清版本
        output_path_hd = output_path.replace('.png', '_HD.png')
        plt.savefig(output_path_hd, dpi=600, bbox_inches='tight',
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"  ✓ 高清版本已保存: {output_path_hd}")
        
        plt.close()
    
    def generate_all_maps(self):
        """生成所有地图"""
        print("=" * 80)
        print("高德地图网格生成器")
        print("=" * 80)
        
        # 加载数据
        print("\n加载数据...")
        chengdu_data = self.load_chengdu_data()
        haikou_data = self.load_haikou_data()
        
        print(f"成都数据: {len(chengdu_data):,} 条")
        print(f"海口数据: {len(haikou_data):,} 条")
        
        # 生成成都地图
        self.generate_city_map(
            '成都',
            self.chengdu_bounds,
            chengdu_data,
            '/root/lanyun-tmp/data_code/chengdu_grid_map.png'
        )
        
        # 生成海口地图
        self.generate_city_map(
            '海口',
            self.haikou_bounds,
            haikou_data,
            '/root/lanyun-tmp/data_code/haikou_grid_map.png'
        )
        
        print("\n" + "=" * 80)
        print("✅ 所有地图生成完成！")
        print("=" * 80)
        
        print("\n使用说明:")
        print("  • 图片无标题、无坐标轴标签，适合作为论文配图")
        print("  • 网格线为蓝色（#0066CC），数据点为红色（#DC143C）")
        print("  • 使用高德地图作为背景，显示真实的地理信息")
        print("  • 范围基于数据集的核心区域（90分位数）")
        print("  • 适合用于论文、演示文稿等学术用途")


def main():
    """主函数"""
    generator = AMapGridGenerator()
    generator.generate_all_maps()


if __name__ == '__main__':
    main()
