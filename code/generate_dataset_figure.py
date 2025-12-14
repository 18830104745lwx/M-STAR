#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成高质量的数据集空间分布与网格划分示意图
适用于学术论文（中英文期刊）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import matplotlib.font_manager as fm
import seaborn as sns
import os
import glob

# ============================================
# SCI一区期刊字体标准配置
# ============================================

# 1. 字体族设置 (学术期刊标准)
plt.rcParams['font.family'] = 'serif'  # 使用serif字体族
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'STSong', 'SimSun', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 专业数学符号字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 2. 分级字体尺寸系统
FONT_SIZES = {
    'title': 14,        # 主标题（醒目但不过大）
    'subtitle': 12,     # 图表标题（清晰可读）
    'axis_label': 11,   # 坐标轴标签（标准轴标签大小）
    'text': 10,         # 基础文本（期刊标准正文大小）
    'tick': 9,          # 刻度标签（坐标轴数值）
    'colorbar': 10,     # 颜色条标签（图例说明）
    'colorbar_tick': 9, # 颜色条刻度（数值刻度）
    'scale': 9,         # 比例尺标签（辅助信息）
    'annotation': 8     # 地理标注（次要标注信息）
}

# 3. 线条粗细标准化
LINE_WIDTHS = {
    'axes': 0.8,        # 坐标轴线（专业标准）
    'grid': 0.5,        # 网格线（辅助参考）
    'default': 1.0,     # 默认线条（图表元素）
    'frame': 0.5        # 边框线（图框轮廓）
}

# 4. 字体粗细规范
FONT_WEIGHTS = {
    'title': 'bold',    # 标题（突出重要性）
    'normal': 'normal', # 坐标轴标签、正文（保持可读性）
    'scale': 'bold'     # 比例尺（重要参考信息）
}

# 5. 自动检测并配置中文字体支持
try:
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    # 优先使用宋体系列（serif风格，符合学术标准）
    chinese_serif_fonts = ['STSong', 'SimSun', 'NSimSun', 'FangSong', 'STFangsong', 
                           'WenQuanYi Micro Hei', 'SimHei']
    for font in chinese_serif_fonts:
        if font in available_fonts:
            # 将中文字体添加到serif列表前面
            current_serif = plt.rcParams['font.serif']
            if font not in current_serif:
                plt.rcParams['font.serif'] = [font] + current_serif
            print(f"✓ 使用中文字体: {font} (Serif风格)")
            break
except Exception as e:
    print(f"警告: 中文字体配置异常 - {e}")

# 6. 全局样式配置
plt.rcParams['axes.linewidth'] = LINE_WIDTHS['axes']       # 坐标轴线宽
plt.rcParams['grid.linewidth'] = LINE_WIDTHS['grid']       # 网格线宽
plt.rcParams['lines.linewidth'] = LINE_WIDTHS['default']   # 默认线宽
plt.rcParams['patch.linewidth'] = LINE_WIDTHS['frame']     # 边框线宽
plt.rcParams['xtick.major.width'] = LINE_WIDTHS['axes']    # X轴刻度宽度
plt.rcParams['ytick.major.width'] = LINE_WIDTHS['axes']    # Y轴刻度宽度
plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']       # X轴刻度字号
plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']       # Y轴刻度字号

print("=" * 60)
print("SCI一区期刊字体标准已配置")
print(f"字体族: Serif (Times New Roman + 中文宋体)")
print(f"数学符号: STIX")
print(f"基础字号: {FONT_SIZES['text']}pt")
print(f"坐标轴线宽: {LINE_WIDTHS['axes']}pt")
print("=" * 60)

class DatasetFigureGenerator:
    """生成数据集可视化图表"""
    
    def __init__(self, data_dir, output_path='dataset_spatial_distribution.png'):
        self.data_dir = data_dir
        self.output_path = output_path
        self.df = None
        
        # 核心城区边界（与chengdu_didi_libcity.py保持一致）
        self.core_bounds = {
            'lon_min': 103.994821,
            'lon_max': 104.134852,
            'lat_min': 30.614351,
            'lat_max': 30.746338
        }
        
        # 网格参数
        self.grid_rows = 20
        self.grid_cols = 20
    
    def load_data(self, sample_ratio=0.1):
        """加载数据（采样以加快处理速度）"""
        print("正在加载数据...")
        
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到CSV文件")
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 读取所有文件
        dfs = []
        for file in csv_files[:5]:  # 只读取前5个文件以加快速度
            df_temp = pd.read_csv(file)
            # 重命名列为英文（方便处理）
            df_temp.columns = ['order_id', 'start_time', 'end_time', 
                              'starting_lng', 'starting_lat', 'dest_lng', 'dest_lat']
            dfs.append(df_temp)
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"原始数据: {len(df)} 条记录")
        
        # 数据清洗（简化版）
        df = df.dropna(subset=['starting_lat', 'starting_lng', 'dest_lat', 'dest_lng'])
        df = df.drop_duplicates(subset=['order_id'])
        
        # 采样（为了可视化速度）
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
            print(f"采样后数据: {len(df)} 条记录")
        
        self.df = df
        return df
    
    def create_figure(self):
        """创建四子图"""
        print("正在生成图表...")
        
        # 创建2x2子图（不添加大标题）
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # (a) 原始数据分布
        self._plot_raw_distribution(axes[0, 0])
        
        # (b) 核心城区提取
        self._plot_core_area_extraction(axes[0, 1])
        
        # (c) 20×20网格划分
        self._plot_grid_division(axes[1, 0])
        
        # (d) 平均流量热力图
        self._plot_flow_heatmap(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {self.output_path}")
        
        return fig
    
    def _plot_raw_distribution(self, ax):
        """(a) 原始数据分布散点图"""
        # 绘制起点分布
        ax.scatter(self.df['starting_lng'], self.df['starting_lat'], 
                  s=1, alpha=0.3, c='#1f77b4', label='上车点', rasterized=True)
        
        # 添加核心区域框
        rect = patches.Rectangle(
            (self.core_bounds['lon_min'], self.core_bounds['lat_min']),
            self.core_bounds['lon_max'] - self.core_bounds['lon_min'],
            self.core_bounds['lat_max'] - self.core_bounds['lat_min'],
            linewidth=LINE_WIDTHS['default'], edgecolor='red', facecolor='none', 
            linestyle='--', label='核心城区范围'
        )
        ax.add_patch(rect)
        
        # 应用SCI标准字体设置
        ax.set_xlabel('经度 (Longitude)', fontsize=FONT_SIZES['axis_label'], 
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('纬度 (Latitude)', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(a) 原始数据分布', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.legend(loc='upper right', fontsize=FONT_SIZES['text'], frameon=True, 
                 edgecolor='gray', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=LINE_WIDTHS['grid'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_core_area_extraction(self, ax):
        """(b) 核心城区提取"""
        # 过滤到核心区域的数据
        core_df = self.df[
            (self.df['starting_lng'] >= self.core_bounds['lon_min']) &
            (self.df['starting_lng'] <= self.core_bounds['lon_max']) &
            (self.df['starting_lat'] >= self.core_bounds['lat_min']) &
            (self.df['starting_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # 绘制核心区域内的点
        ax.scatter(core_df['starting_lng'], core_df['starting_lat'],
                  s=2, alpha=0.4, c='#2ca02c', label='核心区域数据', rasterized=True)
        
        # 添加边界框
        rect = patches.Rectangle(
            (self.core_bounds['lon_min'], self.core_bounds['lat_min']),
            self.core_bounds['lon_max'] - self.core_bounds['lon_min'],
            self.core_bounds['lat_max'] - self.core_bounds['lat_min'],
            linewidth=LINE_WIDTHS['default']*1.5, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加尺寸标注（应用SCI标准字体）
        ax.text(0.05, 0.95, f'区域大小: ~14 km × 14 km\n数据点数: {len(core_df):,}',
                transform=ax.transAxes, fontsize=FONT_SIZES['scale'],
                fontweight=FONT_WEIGHTS['scale'], verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, 
                         edgecolor='gray', linewidth=LINE_WIDTHS['frame']))
        
        # 应用SCI标准字体设置
        ax.set_xlabel('经度 (Longitude)', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('纬度 (Latitude)', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(b) 核心城区提取 (90%分位数)', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.set_xlim(self.core_bounds['lon_min'] - 0.01, self.core_bounds['lon_max'] + 0.01)
        ax.set_ylim(self.core_bounds['lat_min'] - 0.01, self.core_bounds['lat_max'] + 0.01)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=LINE_WIDTHS['grid'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_grid_division(self, ax):
        """(c) 20×20网格划分"""
        # 计算网格线
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        # 过滤核心区域数据
        core_df = self.df[
            (self.df['starting_lng'] >= self.core_bounds['lon_min']) &
            (self.df['starting_lng'] <= self.core_bounds['lon_max']) &
            (self.df['starting_lat'] >= self.core_bounds['lat_min']) &
            (self.df['starting_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # 绘制数据点（半透明）
        ax.scatter(core_df['starting_lng'], core_df['starting_lat'],
                  s=1, alpha=0.15, c='gray', rasterized=True)
        
        # 绘制网格线（应用SCI标准线宽）
        for lon in lon_edges:
            ax.axvline(lon, color='red', linewidth=LINE_WIDTHS['axes'], alpha=0.7)
        for lat in lat_edges:
            ax.axhline(lat, color='red', linewidth=LINE_WIDTHS['axes'], alpha=0.7)
        
        # 标注网格数量（应用SCI标准字体）
        ax.text(0.05, 0.95, f'网格规模: {self.grid_rows} × {self.grid_cols}\n'
                            f'总网格数: {self.grid_rows * self.grid_cols}\n'
                            f'网格分辨率: ~700 m',
                transform=ax.transAxes, fontsize=FONT_SIZES['scale'],
                fontweight=FONT_WEIGHTS['scale'], verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85,
                         edgecolor='gray', linewidth=LINE_WIDTHS['frame']))
        
        # 应用SCI标准字体设置
        ax.set_xlabel('经度 (Longitude)', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('纬度 (Latitude)', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(c) 20×20网格划分', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.set_xlim(self.core_bounds['lon_min'], self.core_bounds['lon_max'])
        ax.set_ylim(self.core_bounds['lat_min'], self.core_bounds['lat_max'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_flow_heatmap(self, ax):
        """(d) 平均流量热力图"""
        # 过滤核心区域数据
        core_df = self.df[
            (self.df['starting_lng'] >= self.core_bounds['lon_min']) &
            (self.df['starting_lng'] <= self.core_bounds['lon_max']) &
            (self.df['starting_lat'] >= self.core_bounds['lat_min']) &
            (self.df['starting_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # 计算网格边界
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        # 统计每个网格的流入量
        flow_matrix = np.zeros((self.grid_rows, self.grid_cols))
        
        for _, row in core_df.iterrows():
            lon_idx = np.searchsorted(lon_edges, row['starting_lng']) - 1
            lat_idx = np.searchsorted(lat_edges, row['starting_lat']) - 1
            
            # 确保索引在有效范围内
            if 0 <= lon_idx < self.grid_cols and 0 <= lat_idx < self.grid_rows:
                flow_matrix[lat_idx, lon_idx] += 1
        
        # 反转纬度方向（使南在下，北在上）
        flow_matrix = np.flipud(flow_matrix)
        
        # 绘制热力图
        im = ax.imshow(flow_matrix, cmap='YlOrRd', aspect='auto', 
                      interpolation='bilinear', alpha=0.9)
        
        # 添加颜色条（应用SCI标准字体）
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('流量计数 (Counts)', fontsize=FONT_SIZES['colorbar'],
                      fontweight=FONT_WEIGHTS['normal'])
        cbar.ax.tick_params(labelsize=FONT_SIZES['colorbar_tick'], 
                           width=LINE_WIDTHS['axes'])
        cbar.outline.set_linewidth(LINE_WIDTHS['frame'])
        
        # 设置刻度
        ax.set_xticks(np.arange(0, self.grid_cols, 5))
        ax.set_yticks(np.arange(0, self.grid_rows, 5))
        ax.set_xticklabels(np.arange(0, self.grid_cols, 5))
        ax.set_yticklabels(np.arange(self.grid_rows, 0, -5))
        
        # 添加统计信息（应用SCI标准字体）
        total_flow = flow_matrix.sum()
        non_zero_cells = np.count_nonzero(flow_matrix)
        avg_flow = flow_matrix[flow_matrix > 0].mean() if non_zero_cells > 0 else 0
        sparsity = (self.grid_rows * self.grid_cols - non_zero_cells) / (self.grid_rows * self.grid_cols) * 100
        
        ax.text(0.05, 0.95, 
                f'总流量: {int(total_flow):,}\n'
                f'非零网格: {non_zero_cells}/{self.grid_rows * self.grid_cols}\n'
                f'平均流量: {avg_flow:.2f}\n'
                f'稀疏率: {sparsity:.2f}%',
                transform=ax.transAxes, fontsize=FONT_SIZES['scale'],
                fontweight=FONT_WEIGHTS['scale'], verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85,
                         edgecolor='gray', linewidth=LINE_WIDTHS['frame']))
        
        # 应用SCI标准字体设置
        ax.set_xlabel('网格列号 (Column ID)', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('网格行号 (Row ID)', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(d) 平均流量热力图', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.grid(False)


def main():
    """主函数"""
    # 设置路径
    data_dir = '/root/lanyun-tmp/Bigscity-LibCity-Datasets-master/Bigscity-LibCity-Datasets-master/data/2016年11月成都网约车滴滴订单数据'
    output_path = '/root/lanyun-tmp/Bigscity-LibCity-Datasets-master/Bigscity-LibCity-Datasets-master/dataset_spatial_distribution.png'
    
    # 创建生成器
    generator = DatasetFigureGenerator(data_dir, output_path)
    
    # 加载数据（采样10%以加快速度）
    generator.load_data(sample_ratio=0.15)
    
    # 生成图表
    fig = generator.create_figure()
    
    print("\n图表生成完成！")
    print(f"图片路径: {output_path}")
    print("图片规格: 300 DPI, 适合直接用于学术论文")
    
    # 同时生成高分辨率版本（600 DPI）
    output_path_hd = output_path.replace('.png', '_HD.png')
    fig.savefig(output_path_hd, dpi=600, bbox_inches='tight')
    print(f"高清版本: {output_path_hd} (600 DPI)")


if __name__ == '__main__':
    main()

