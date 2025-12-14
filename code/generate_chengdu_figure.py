#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Spatial Distribution and Grid Partition Visualization for Chengdu DiDi Dataset
Suitable for International Academic Publications
Similar style to Haikou dataset visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import seaborn as sns
import os
import glob

# ============================================
# Academic Journal Standard Font Configuration
# ============================================

# Font family settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# Font size system
FONT_SIZES = {
    'title': 16,
    'subtitle': 13,
    'axis_label': 12,
    'text': 11,
    'tick': 10,
    'legend': 10,
    'annotation': 9
}

# Line width standards
LINE_WIDTHS = {
    'axes': 1.0,
    'grid': 0.6,
    'default': 1.2,
    'frame': 0.8
}

# Global style configuration
plt.rcParams['axes.linewidth'] = LINE_WIDTHS['axes']
plt.rcParams['grid.linewidth'] = LINE_WIDTHS['grid']
plt.rcParams['lines.linewidth'] = LINE_WIDTHS['default']
plt.rcParams['patch.linewidth'] = LINE_WIDTHS['frame']
plt.rcParams['xtick.major.width'] = LINE_WIDTHS['axes']
plt.rcParams['ytick.major.width'] = LINE_WIDTHS['axes']
plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']
plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']

print("=" * 80)
print("Academic Journal Standard Configuration Loaded")
print(f"Font: Serif (Times New Roman)")
print(f"Math Symbols: STIX")
print(f"Base Font Size: {FONT_SIZES['text']}pt")
print("=" * 80)


# ============================================
# Custom Color Schemes (Similar to Haikou)
# ============================================

COLOR_SCHEMES = {
    'primary': '#2E86AB',      # Deep Blue
    'secondary': '#A23B72',    # Purple-Red
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Teal Green
    'warning': '#D84315',      # Deep Orange-Red
    'neutral': '#5E6472',      # Gray-Blue
    'highlight': '#C73E1D',    # Brick Red
    'light_bg': '#F5F5F5'      # Light Gray
}

# Custom colormaps
HEATMAP_CMAP = 'hot'  # Red-Yellow color scheme for better visibility
SCATTER_ALPHA = 0.35


class ChengduDatasetFigureGenerator:
    """Generate visualization figures for Chengdu DiDi dataset"""
    
    def __init__(self, data_dir, output_path='chengdu_spatial_distribution.png'):
        self.data_dir = data_dir
        self.output_path = output_path
        self.df = None
        
        # Chengdu core urban area boundaries (from processed data)
        # Longitude: [103.997131, 104.137028], Latitude: [30.614391, 30.739670]
        # Coverage: 13.50 km × 13.91 km
        self.core_bounds = {
            'lon_min': 103.997131,
            'lon_max': 104.137028,
            'lat_min': 30.614391,
            'lat_max': 30.739670
        }
        
        # Grid parameters
        self.grid_rows = 20
        self.grid_cols = 20
    
    def load_data(self, sample_ratio=0.10, max_files=3):
        """Load and preprocess DiDi taxi data"""
        print("Loading Chengdu DiDi dataset...")
        
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load multiple files for better representation
        dfs = []
        for file in csv_files[:max_files]:
            try:
                df_temp = pd.read_csv(file)
                print(f"  Loaded: {os.path.basename(file)} ({len(df_temp):,} records)")
                
                # Rename columns to English
                df_temp = df_temp.rename(columns={
                    '订单ID': 'order_id',
                    '开始计费时间': 'start_time',
                    '结束计费时间': 'end_time',
                    '上车位置经度': 'start_lng',
                    '上车位置纬度': 'start_lat',
                    '下车位置经度': 'end_lng',
                    '下车位置纬度': 'end_lat'
                })
                
                dfs.append(df_temp)
            except Exception as e:
                print(f"  Warning: Failed to load {file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("Failed to load any data files")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Raw data: {len(df):,} records")
        
        # Data cleaning
        required_cols = ['start_lng', 'start_lat', 'end_lng', 'end_lat']
        df = df.dropna(subset=required_cols)
        
        # Filter to reasonable geographic range (Chengdu area)
        df = df[
            (df['start_lng'] >= 103.5) & (df['start_lng'] <= 104.5) &
            (df['start_lat'] >= 30.3) & (df['start_lat'] <= 31.0)
        ]
        print(f"After cleaning: {len(df):,} records")
        
        # Sampling for visualization
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
            print(f"After sampling ({sample_ratio*100:.0f}%): {len(df):,} records")
        
        self.df = df
        return df
    
    def create_figure(self):
        """Create comprehensive visualization with 4 subplots"""
        print("\nGenerating visualization figures...")
        
        # Create 2x2 subplot layout
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
        
        axes = [
            fig.add_subplot(gs[0, 0]),  # (a) Full dataset distribution
            fig.add_subplot(gs[0, 1]),  # (b) Core area selection
            fig.add_subplot(gs[1, 0]),  # (c) Grid partition system
            fig.add_subplot(gs[1, 1])   # (d) Flow density heatmap
        ]
        
        # Generate each subplot
        self._plot_full_distribution(axes[0])
        self._plot_core_extraction(axes[1])
        self._plot_grid_system(axes[2])
        self._plot_density_heatmap(axes[3])
        
        # Save in multiple resolutions
        plt.savefig(self.output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Standard version saved: {self.output_path}")
        
        output_path_hd = self.output_path.replace('.png', '_HD.png')
        plt.savefig(output_path_hd, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✓ High-resolution version saved: {output_path_hd}")
        
        output_path_small = self.output_path.replace('.png', '_preview.png')
        plt.savefig(output_path_small, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Preview version saved: {output_path_small}")
        
        plt.close()
        return fig
    
    def _plot_full_distribution(self, ax):
        """(a) Full dataset spatial distribution"""
        # Plot pickup locations with custom color
        ax.scatter(self.df['start_lng'], self.df['start_lat'], 
                  s=0.8, alpha=SCATTER_ALPHA, c=COLOR_SCHEMES['primary'], 
                  label='Pickup locations', rasterized=True, edgecolors='none')
        
        # Add core area rectangle
        rect = patches.Rectangle(
            (self.core_bounds['lon_min'], self.core_bounds['lat_min']),
            self.core_bounds['lon_max'] - self.core_bounds['lon_min'],
            self.core_bounds['lat_max'] - self.core_bounds['lat_min'],
            linewidth=LINE_WIDTHS['default']*1.5, 
            edgecolor=COLOR_SCHEMES['warning'], 
            facecolor='none', 
            linestyle='--', 
            label='Core urban area'
        )
        ax.add_patch(rect)
        
        # Styling
        ax.set_xlabel('Longitude (°E)', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_ylabel('Latitude (°N)', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_title('(a) Full Dataset Distribution', 
                    fontsize=FONT_SIZES['subtitle'], fontweight='bold', pad=12)
        ax.legend(loc='upper left', fontsize=FONT_SIZES['legend'], 
                 frameon=True, shadow=True, fancybox=True)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=LINE_WIDTHS['grid'])
        ax.set_aspect('equal', adjustable='box')
        
        # Add data statistics
        stats_text = f'Total trips: {len(self.df):,}\nArea: Chengdu City'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=FONT_SIZES['annotation'], ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        alpha=0.8, edgecolor='gray'))
    
    def _plot_core_extraction(self, ax):
        """(b) Core area extraction (90th percentile)"""
        # Filter to core area
        core_df = self.df[
            (self.df['start_lng'] >= self.core_bounds['lon_min']) &
            (self.df['start_lng'] <= self.core_bounds['lon_max']) &
            (self.df['start_lat'] >= self.core_bounds['lat_min']) &
            (self.df['start_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # Plot with different color scheme
        ax.scatter(core_df['start_lng'], core_df['start_lat'],
                  s=1.5, alpha=0.45, c=COLOR_SCHEMES['success'], 
                  label='Core area trips', rasterized=True, edgecolors='none')
        
        # Add boundary box with different style
        rect = patches.Rectangle(
            (self.core_bounds['lon_min'], self.core_bounds['lat_min']),
            self.core_bounds['lon_max'] - self.core_bounds['lon_min'],
            self.core_bounds['lat_max'] - self.core_bounds['lat_min'],
            linewidth=LINE_WIDTHS['default']*2, 
            edgecolor=COLOR_SCHEMES['highlight'], 
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)
        
        # Calculate actual coverage (Chengdu at ~30°N)
        lon_span_km = (self.core_bounds['lon_max'] - self.core_bounds['lon_min']) * 96.5
        lat_span_km = (self.core_bounds['lat_max'] - self.core_bounds['lat_min']) * 111.0
        
        # Add annotation box
        info_text = (f'Coverage: {lon_span_km:.1f} km × {lat_span_km:.1f} km\n'
                    f'Data points: {len(core_df):,}\n'
                    f'Retention: {len(core_df)/len(self.df)*100:.1f}%')
        ax.text(0.03, 0.97, info_text, transform=ax.transAxes, 
               fontsize=FONT_SIZES['annotation'], va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                        alpha=0.9, edgecolor='orange', linewidth=1.5))
        
        # Styling
        ax.set_xlabel('Longitude (°E)', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_ylabel('Latitude (°N)', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_title('(b) Core Urban Area Extraction', 
                    fontsize=FONT_SIZES['subtitle'], fontweight='bold', pad=12)
        ax.set_xlim(self.core_bounds['lon_min'] - 0.015, self.core_bounds['lon_max'] + 0.015)
        ax.set_ylim(self.core_bounds['lat_min'] - 0.015, self.core_bounds['lat_max'] + 0.015)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=LINE_WIDTHS['grid'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_grid_system(self, ax):
        """(c) Grid partition system (20×20)"""
        # Calculate grid edges
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        # Filter core area data
        core_df = self.df[
            (self.df['start_lng'] >= self.core_bounds['lon_min']) &
            (self.df['start_lng'] <= self.core_bounds['lon_max']) &
            (self.df['start_lat'] >= self.core_bounds['lat_min']) &
            (self.df['start_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # Plot data points with different style
        ax.scatter(core_df['start_lng'], core_df['start_lat'],
                  s=0.8, alpha=0.2, c=COLOR_SCHEMES['neutral'], 
                  rasterized=True, edgecolors='none')
        
        # Draw grid lines with custom color
        for lon in lon_edges:
            ax.axvline(lon, color=COLOR_SCHEMES['accent'], 
                      linewidth=LINE_WIDTHS['axes']*0.9, alpha=0.75)
        for lat in lat_edges:
            ax.axhline(lat, color=COLOR_SCHEMES['accent'], 
                      linewidth=LINE_WIDTHS['axes']*0.9, alpha=0.75)
        
        # Calculate grid resolution
        lon_span_km = (self.core_bounds['lon_max'] - self.core_bounds['lon_min']) * 96.5
        lat_span_km = (self.core_bounds['lat_max'] - self.core_bounds['lat_min']) * 111.0
        grid_res_km = np.mean([lon_span_km / self.grid_cols, lat_span_km / self.grid_rows])
        
        # Add grid information
        grid_info = (f'Grid size: {self.grid_rows} × {self.grid_cols}\n'
                    f'Total cells: {self.grid_rows * self.grid_cols}\n'
                    f'Resolution: ~{grid_res_km:.2f} km/cell')
        ax.text(0.03, 0.97, grid_info, transform=ax.transAxes, 
               fontsize=FONT_SIZES['annotation'], va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='lightcyan', 
                        alpha=0.9, edgecolor='steelblue', linewidth=1.5))
        
        # Styling
        ax.set_xlabel('Longitude (°E)', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_ylabel('Latitude (°N)', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_title('(c) Grid Partition System (20×20)', 
                    fontsize=FONT_SIZES['subtitle'], fontweight='bold', pad=12)
        ax.set_xlim(self.core_bounds['lon_min'], self.core_bounds['lon_max'])
        ax.set_ylim(self.core_bounds['lat_min'], self.core_bounds['lat_max'])
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_density_heatmap(self, ax):
        """(d) Spatial density heatmap"""
        # Filter core area data
        core_df = self.df[
            (self.df['start_lng'] >= self.core_bounds['lon_min']) &
            (self.df['start_lng'] <= self.core_bounds['lon_max']) &
            (self.df['start_lat'] >= self.core_bounds['lat_min']) &
            (self.df['start_lat'] <= self.core_bounds['lat_max'])
        ]
        
        # Calculate grid boundaries
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        # Compute density matrix
        density_matrix = np.zeros((self.grid_rows, self.grid_cols))
        
        for _, row in core_df.iterrows():
            lon_idx = np.searchsorted(lon_edges, row['start_lng']) - 1
            lat_idx = np.searchsorted(lat_edges, row['start_lat']) - 1
            
            # Ensure valid indices
            if 0 <= lon_idx < self.grid_cols and 0 <= lat_idx < self.grid_rows:
                density_matrix[lat_idx, lon_idx] += 1
        
        # Flip latitude for correct orientation
        density_matrix = np.flipud(density_matrix)
        
        # Plot heatmap with hot colormap
        im = ax.imshow(density_matrix, cmap=HEATMAP_CMAP, aspect='auto', 
                      interpolation='gaussian', alpha=0.95)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Trip Density (counts)', fontsize=FONT_SIZES['legend'],
                      fontweight='normal')
        cbar.ax.tick_params(labelsize=FONT_SIZES['tick'], 
                           width=LINE_WIDTHS['axes'])
        cbar.outline.set_linewidth(LINE_WIDTHS['frame'])
        
        # Set ticks
        ax.set_xticks(np.arange(0, self.grid_cols, 4))
        ax.set_yticks(np.arange(0, self.grid_rows, 4))
        ax.set_xticklabels(np.arange(0, self.grid_cols, 4))
        ax.set_yticklabels(np.arange(self.grid_rows, 0, -4))
        
        # Calculate statistics
        total_trips = int(density_matrix.sum())
        active_cells = np.count_nonzero(density_matrix)
        mean_density = density_matrix[density_matrix > 0].mean() if active_cells > 0 else 0
        max_density = density_matrix.max()
        sparsity_rate = (self.grid_rows * self.grid_cols - active_cells) / (self.grid_rows * self.grid_cols) * 100
        
        # Add statistics box
        stats_info = (f'Total trips: {total_trips:,}\n'
                     f'Active cells: {active_cells}/{self.grid_rows * self.grid_cols}\n'
                     f'Mean density: {mean_density:.1f}\n'
                     f'Max density: {int(max_density):,}\n'
                     f'Sparsity: {sparsity_rate:.1f}%')
        ax.text(0.03, 0.97, stats_info, transform=ax.transAxes, 
               fontsize=FONT_SIZES['annotation'], va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', 
                        alpha=0.9, edgecolor='green', linewidth=1.5))
        
        # Styling
        ax.set_xlabel('Grid Column Index', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_ylabel('Grid Row Index', fontsize=FONT_SIZES['axis_label'], fontweight='normal')
        ax.set_title('(d) Spatial Density Heatmap', 
                    fontsize=FONT_SIZES['subtitle'], fontweight='bold', pad=12)
        ax.grid(False)


def main():
    """Main execution function"""
    print("=" * 80)
    print("Chengdu DiDi Dataset Visualization Generator")
    print("=" * 80)
    print()
    
    # Set paths
    data_dir = '/root/lanyun-tmp/data_code/raw_data/2016年11月成都网约车滴滴订单数据'
    output_path = '/root/lanyun-tmp/data_code/chengdu_spatial_distribution.png'
    
    # Create generator
    generator = ChengduDatasetFigureGenerator(data_dir, output_path)
    
    try:
        # Load data (sample 10% for visualization, use first 3 files)
        generator.load_data(sample_ratio=0.10, max_files=3)
        
        # Generate figure
        fig = generator.create_figure()
        
        print()
        print("=" * 80)
        print("✓ Figure generation completed successfully!")
        print("=" * 80)
        print(f"Standard version (300 DPI): {output_path}")
        print(f"High-resolution (600 DPI): {output_path.replace('.png', '_HD.png')}")
        print(f"Preview (150 DPI): {output_path.replace('.png', '_preview.png')}")
        print()
        print("Specifications:")
        print("  • Language: Full English (suitable for international journals)")
        print("  • Font: Times New Roman (academic standard)")
        print("  • Color scheme: Consistent with Haikou dataset")
        print("  • Layout: Optimized 2×2 grid")
        print("  • Heatmap: Hot colormap (black-red-yellow-white)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

