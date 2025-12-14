#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate high-quality dataset spatial distribution and grid partition visualization
Suitable for academic papers (Chinese/English journals)
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
# SCI Tier 1 Journal Font Standard Configuration
# ============================================

# 1. Font family settings (academic journal standards)
plt.rcParams['font.family'] = 'serif'  # Use serif font family
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'STSong', 'SimSun', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # Professional math symbol font
plt.rcParams['axes.unicode_minus'] = False  # Solve minus sign display issue

# 2. Hierarchical font size system
FONT_SIZES = {
    'title': 14,        # Main title (prominent but not too large)
    'subtitle': 12,     # Chart title (clear and readable)
    'axis_label': 11,   # Axis labels (standard axis label size)
    'text': 10,         # Base text (journal standard body text size)
    'tick': 9,          # Tick labels (axis numerical values)
    'colorbar': 10,     # Color bar labels (legend description)
    'colorbar_tick': 9, # Color bar ticks (numerical scales)
    'scale': 9,         # Scale bar labels (auxiliary information)
    'annotation': 8     # Geographic annotations (secondary annotation info)
}

# 3. Line width standardization
LINE_WIDTHS = {
    'axes': 0.8,        # Axis lines (professional standard)
    'grid': 0.5,        # Grid lines (auxiliary reference)
    'default': 1.0,     # Default lines (chart elements)
    'frame': 0.5        # Border lines (frame outline)
}

# 4. Font weight specification
FONT_WEIGHTS = {
    'title': 'bold',    # Title (emphasize importance)
    'normal': 'normal', # Axis labels, body text (maintain readability)
    'scale': 'bold'     # Scale (important reference information)
}

# 5. Automatically detect and configure Chinese font support
try:
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    # Prefer Songti series (serif style, meets academic standards)
    chinese_serif_fonts = ['STSong', 'SimSun', 'NSimSun', 'FangSong', 'STFangsong', 
                           'WenQuanYi Micro Hei', 'SimHei']
    for font in chinese_serif_fonts:
        if font in available_fonts:
            # Add Chinese font to the beginning of serif list
            current_serif = plt.rcParams['font.serif']
            if font not in current_serif:
                plt.rcParams['font.serif'] = [font] + current_serif
            print(f"✓ Using Chinese font: {font} (Serif style)")
            break
except Exception as e:
    print(f"Warning: Chinese font configuration exception - {e}")

# 6. Global style configuration
plt.rcParams['axes.linewidth'] = LINE_WIDTHS['axes']       # Axis line width
plt.rcParams['grid.linewidth'] = LINE_WIDTHS['grid']       # Grid line width
plt.rcParams['lines.linewidth'] = LINE_WIDTHS['default']   # Default line width
plt.rcParams['patch.linewidth'] = LINE_WIDTHS['frame']     # Border line width
plt.rcParams['xtick.major.width'] = LINE_WIDTHS['axes']    # X-axis tick width
plt.rcParams['ytick.major.width'] = LINE_WIDTHS['axes']    # Y-axis tick width
plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']       # X-axis tick font size
plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']       # Y-axis tick font size

print("=" * 60)
print("SCI Tier 1 Journal Font Standards Configured")
print(f"Font Family: Serif (Times New Roman + Chinese Songti)")
print(f"Math Symbols: STIX")
print(f"Base Font Size: {FONT_SIZES['text']}pt")
print(f"Axis Line Width: {LINE_WIDTHS['axes']}pt")
print("=" * 60)

class DatasetFigureGenerator:
    """Generate dataset visualization charts"""
    
    def __init__(self, data_dir, output_path='dataset_spatial_distribution.png'):
        self.data_dir = data_dir
        self.output_path = output_path
        self.df = None
        
        # Core urban area boundaries (consistent with chengdu_didi_libcity.py)
        self.core_bounds = {
            'lon_min': 103.994821,
            'lon_max': 104.134852,
            'lat_min': 30.614351,
            'lat_max': 30.746338
        }
        
        # Grid parameters
        self.grid_rows = 20
        self.grid_cols = 20
    
    def load_data(self, sample_ratio=0.1):
        """Load data (sampled to speed up processing)"""
        print("Loading data...")
        
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Read all files
        dfs = []
        for file in csv_files[:5]:  # Only read first 5 files for speed
            df_temp = pd.read_csv(file)
            # Rename columns to English (for easier processing)
            df_temp.columns = ['order_id', 'start_time', 'end_time', 
                              'starting_lng', 'starting_lat', 'dest_lng', 'dest_lat']
            dfs.append(df_temp)
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Raw data: {len(df)} records")
        
        # Data cleaning (simplified version)
        df = df.dropna(subset=['starting_lat', 'starting_lng', 'dest_lat', 'dest_lng'])
        df = df.drop_duplicates(subset=['order_id'])
        
        # Sampling (for visualization speed)
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
            print(f"Sampled data: {len(df)} records")
        
        self.df = df
        return df
    
    def create_figure(self):
        """Create 2x2 subplot figure"""
        print("Generating chart...")
        
        # Create 2x2 subplot (no main title)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # (a) Raw data distribution
        self._plot_raw_distribution(axes[0, 0])
        
        # (b) Core area extraction
        self._plot_core_area_extraction(axes[0, 1])
        
        # (c) 20×20 grid division
        self._plot_grid_division(axes[1, 0])
        
        # (d) Average flow heatmap
        self._plot_flow_heatmap(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {self.output_path}")
        
        return fig
    
    def _plot_raw_distribution(self, ax):
        """(a) Raw data distribution scatter plot"""
        # Plot origin point distribution
        ax.scatter(self.df['starting_lng'], self.df['starting_lat'], 
                  s=1, alpha=0.3, c='#1f77b4', label='Pickup Points', rasterized=True)
        
        # Add core area rectangle
        rect = patches.Rectangle(
            (self.core_bounds['lon_min'], self.core_bounds['lat_min']),
            self.core_bounds['lon_max'] - self.core_bounds['lon_min'],
            self.core_bounds['lat_max'] - self.core_bounds['lat_min'],
            linewidth=LINE_WIDTHS['default'], edgecolor='red', facecolor='none', 
            linestyle='--', label='Core Urban Area'
        )
        ax.add_patch(rect)
        
        # Apply SCI standard font settings
        ax.set_xlabel('Longitude', fontsize=FONT_SIZES['axis_label'], 
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('Latitude', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(a) Raw Data Distribution', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.legend(loc='upper right', fontsize=FONT_SIZES['text'], frameon=True, 
                 edgecolor='gray', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=LINE_WIDTHS['grid'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_core_area_extraction(self, ax):
        """(b) Core area extraction"""
        # Filter data to core area
        core_df = self.df[
            (self.df['starting_lng'] >= self.core_bounds['lon_min']) &
            (self.df['starting_lng'] <= self.core_bounds['lon_max']) &
            (self.df['starting_lat'] >= self.core_bounds['lat_min']) &
            (self.df['starting_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # Plot points within core area
        ax.scatter(core_df['starting_lng'], core_df['starting_lat'],
                  s=2, alpha=0.4, c='#2ca02c', label='Core Area Data', rasterized=True)
        
        # Add boundary rectangle
        rect = patches.Rectangle(
            (self.core_bounds['lon_min'], self.core_bounds['lat_min']),
            self.core_bounds['lon_max'] - self.core_bounds['lon_min'],
            self.core_bounds['lat_max'] - self.core_bounds['lat_min'],
            linewidth=LINE_WIDTHS['default']*1.5, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add size annotation (apply SCI standard font)
        ax.text(0.05, 0.95, f'Area Size: ~14 km × 14 km\nData Points: {len(core_df):,}',
                transform=ax.transAxes, fontsize=FONT_SIZES['scale'],
                fontweight=FONT_WEIGHTS['scale'], verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, 
                         edgecolor='gray', linewidth=LINE_WIDTHS['frame']))
        
        # Apply SCI standard font settings
        ax.set_xlabel('Longitude', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('Latitude', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(b) Core Urban Area Extraction (90th Percentile)', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.set_xlim(self.core_bounds['lon_min'] - 0.01, self.core_bounds['lon_max'] + 0.01)
        ax.set_ylim(self.core_bounds['lat_min'] - 0.01, self.core_bounds['lat_max'] + 0.01)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=LINE_WIDTHS['grid'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_grid_division(self, ax):
        """(c) 20×20 grid division"""
        # Calculate grid lines
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        # Filter core area data
        core_df = self.df[
            (self.df['starting_lng'] >= self.core_bounds['lon_min']) &
            (self.df['starting_lng'] <= self.core_bounds['lon_max']) &
            (self.df['starting_lat'] >= self.core_bounds['lat_min']) &
            (self.df['starting_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # Plot data points (semi-transparent)
        ax.scatter(core_df['starting_lng'], core_df['starting_lat'],
                  s=1, alpha=0.15, c='gray', rasterized=True)
        
        # Draw grid lines (apply SCI standard line width)
        for lon in lon_edges:
            ax.axvline(lon, color='red', linewidth=LINE_WIDTHS['axes'], alpha=0.7)
        for lat in lat_edges:
            ax.axhline(lat, color='red', linewidth=LINE_WIDTHS['axes'], alpha=0.7)
        
        # Annotate grid count (apply SCI standard font)
        ax.text(0.05, 0.95, f'Grid Scale: {self.grid_rows} × {self.grid_cols}\n'
                            f'Total Grids: {self.grid_rows * self.grid_cols}\n'
                            f'Grid Resolution: ~700 m',
                transform=ax.transAxes, fontsize=FONT_SIZES['scale'],
                fontweight=FONT_WEIGHTS['scale'], verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85,
                         edgecolor='gray', linewidth=LINE_WIDTHS['frame']))
        
        # Apply SCI standard font settings
        ax.set_xlabel('Longitude', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('Latitude', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(c) 20×20 Grid Division', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.set_xlim(self.core_bounds['lon_min'], self.core_bounds['lon_max'])
        ax.set_ylim(self.core_bounds['lat_min'], self.core_bounds['lat_max'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_flow_heatmap(self, ax):
        """(d) Average flow heatmap"""
        # Filter core area data
        core_df = self.df[
            (self.df['starting_lng'] >= self.core_bounds['lon_min']) &
            (self.df['starting_lng'] <= self.core_bounds['lon_max']) &
            (self.df['starting_lat'] >= self.core_bounds['lat_min']) &
            (self.df['starting_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # Calculate grid boundaries
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        # Count inflow for each grid
        flow_matrix = np.zeros((self.grid_rows, self.grid_cols))
        
        for _, row in core_df.iterrows():
            lon_idx = np.searchsorted(lon_edges, row['starting_lng']) - 1
            lat_idx = np.searchsorted(lat_edges, row['starting_lat']) - 1
            
            # Ensure index is within valid range
            if 0 <= lon_idx < self.grid_cols and 0 <= lat_idx < self.grid_rows:
                flow_matrix[lat_idx, lon_idx] += 1
        
        # Reverse latitude direction (south at bottom, north at top)
        flow_matrix = np.flipud(flow_matrix)
        
        # Plot heatmap
        im = ax.imshow(flow_matrix, cmap='YlOrRd', aspect='auto', 
                      interpolation='bilinear', alpha=0.9)
        
        # Add color bar (apply SCI standard font)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Flow Count', fontsize=FONT_SIZES['colorbar'],
                      fontweight=FONT_WEIGHTS['normal'])
        cbar.ax.tick_params(labelsize=FONT_SIZES['colorbar_tick'], 
                           width=LINE_WIDTHS['axes'])
        cbar.outline.set_linewidth(LINE_WIDTHS['frame'])
        
        # Set ticks
        ax.set_xticks(np.arange(0, self.grid_cols, 5))
        ax.set_yticks(np.arange(0, self.grid_rows, 5))
        ax.set_xticklabels(np.arange(0, self.grid_cols, 5))
        ax.set_yticklabels(np.arange(self.grid_rows, 0, -5))
        
        # Add statistics (apply SCI standard font)
        total_flow = flow_matrix.sum()
        non_zero_cells = np.count_nonzero(flow_matrix)
        avg_flow = flow_matrix[flow_matrix > 0].mean() if non_zero_cells > 0 else 0
        sparsity = (self.grid_rows * self.grid_cols - non_zero_cells) / (self.grid_rows * self.grid_cols) * 100
        
        ax.text(0.05, 0.95, 
                f'Total Flow: {int(total_flow):,}\n'
                f'Non-zero Grids: {non_zero_cells}/{self.grid_rows * self.grid_cols}\n'
                f'Avg Flow: {avg_flow:.2f}\n'
                f'Sparsity: {sparsity:.2f}%',
                transform=ax.transAxes, fontsize=FONT_SIZES['scale'],
                fontweight=FONT_WEIGHTS['scale'], verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85,
                         edgecolor='gray', linewidth=LINE_WIDTHS['frame']))
        
        # Apply SCI standard font settings
        ax.set_xlabel('Grid Column ID', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_ylabel('Grid Row ID', fontsize=FONT_SIZES['axis_label'],
                     fontweight=FONT_WEIGHTS['normal'])
        ax.set_title('(d) Average Flow Heatmap', fontsize=FONT_SIZES['subtitle'], 
                    fontweight=FONT_WEIGHTS['title'], pad=10)
        ax.grid(False)


def main():
    """Main function"""
    # Set paths
    data_dir = '/root/lanyun-tmp/Bigscity-LibCity-Datasets-master/Bigscity-LibCity-Datasets-master/data/2016年11月成都网约车滴滴订单数据'
    output_path = '/root/lanyun-tmp/Bigscity-LibCity-Datasets-master/Bigscity-LibCity-Datasets-master/dataset_spatial_distribution.png'
    
    # Create generator
    generator = DatasetFigureGenerator(data_dir, output_path)
    
    # Load data (sampled 10% for speed)
    generator.load_data(sample_ratio=0.15)
    
    # Generate chart
    fig = generator.create_figure()
    
    print("\nChart generation completed!")
    print(f"Image path: {output_path}")
    print("Image specification: 300 DPI, suitable for direct use in academic papers")
    
    # Also generate high-resolution version (600 DPI)
    output_path_hd = output_path.replace('.png', '_HD.png')
    fig.savefig(output_path_hd, dpi=600, bbox_inches='tight')
    print(f"High-definition version: {output_path_hd} (600 DPI)")


if __name__ == '__main__':
    main()
