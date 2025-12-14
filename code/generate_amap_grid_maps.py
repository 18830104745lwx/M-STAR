#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate grid maps with AMap background (Chengdu and Haikou)
Using English version of AMap tile service
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

# Set matplotlib parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False

class AMapGridGenerator:
    """Generate grid maps with AMap background"""
    
    def __init__(self):
        # Chengdu core urban area boundaries
        self.chengdu_bounds = {
            'lon_min': 103.994821,
            'lon_max': 104.134852,
            'lat_min': 30.614351,
            'lat_max': 30.746338
        }
        
        # Haikou core urban area boundaries
        self.haikou_bounds = {
            'lon_min': 110.277127,
            'lon_max': 110.376339,
            'lat_min': 19.978258,
            'lat_max': 20.060804
        }
        
        # Grid parameters
        self.grid_rows = 20
        self.grid_cols = 20
        
        # AMap tile service URLs
        # style=7 vector map, style=8 road map, style=6 satellite
        self.amap_urls = [
            # AMap vector map (standard version)
            'http://webrd01.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            'http://webrd02.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            'http://webrd03.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            'http://webrd04.is.autonavi.com/appmaptile?size=1&scale=1&style=7&x={x}&y={y}&z={z}',
            # Backup: AMap road map
            'http://webrd01.is.autonavi.com/appmaptile?size=1&scale=1&style=8&x={x}&y={y}&z={z}',
            # Backup: CartoDB light map
            'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
            # Backup: OpenStreetMap
            'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        ]
    
    def load_chengdu_data(self, sample_ratio=0.1):
        """Load Chengdu data"""
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
        """Load Haikou data"""
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
        """Add AMap basemap"""
        # Set axis limits (must be set before adding basemap)
        ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
        ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
        
        basemap_added = False
        
        print("  Attempting to load AMap...")
        
        # Try different map sources
        map_names = [
            "AMap Vector 1",
            "AMap Vector 2", 
            "AMap Vector 3",
            "AMap Vector 4",
            "AMap Road Map",
            "CartoDB Light Map",
            "OpenStreetMap"
        ]
        
        for i, url in enumerate(self.amap_urls):
            try:
                if i < len(map_names):
                    print(f"    Trying {map_names[i]}...")
                
                # Use contextily to add custom tile map
                ctx.add_basemap(
                    ax,
                    crs='EPSG:4326',
                    source=url,
                    zoom=12,  # Moderate zoom level
                    alpha=0.9,
                    attribution=""
                )
                
                print(f"    ✓ Successfully loaded {map_names[i] if i < len(map_names) else 'map'}")
                basemap_added = True
                break
                
            except Exception as e:
                continue
        
        # If all online maps fail, use fallback background
        if not basemap_added:
            print("  Warning: All map sources failed, using default background")
            # Create simple gray background
            ax.set_facecolor('#f5f5f5')
            
            # Add simple grid as background
            lon_range = bounds['lon_max'] - bounds['lon_min']
            lat_range = bounds['lat_max'] - bounds['lat_min']
            
            # Background fine grid
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
        """Generate map for a single city"""
        print(f"\nGenerating {city_name} map...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Add AMap background
        self.add_amap_basemap(ax, bounds)
        
        # Plot data points (red)
        if len(data) > 0:
            if city_name == 'Chengdu':
                ax.scatter(data['lng'], data['lat'],
                          s=8, alpha=0.7, c='#DC143C', edgecolors='white', 
                          linewidth=0.3, rasterized=True, zorder=5)
            else:  # Haikou
                ax.scatter(data['starting_wgs84_lng'], data['starting_wgs84_lat'],
                          s=8, alpha=0.7, c='#DC143C', edgecolors='white', 
                          linewidth=0.3, rasterized=True, zorder=5)
            print(f"  ✓ Plotted {len(data):,} data points")
        
        # Create grid boundaries
        lon_edges = np.linspace(bounds['lon_min'], bounds['lon_max'], self.grid_cols + 1)
        lat_edges = np.linspace(bounds['lat_min'], bounds['lat_max'], self.grid_rows + 1)
        
        # Draw grid lines (blue, bold)
        for lon in lon_edges:
            ax.axvline(lon, color='#0066CC', linewidth=2, alpha=0.9, zorder=6)
        for lat in lat_edges:
            ax.axhline(lat, color='#0066CC', linewidth=2, alpha=0.9, zorder=6)
        print(f"  ✓ Drawn {self.grid_rows}×{self.grid_cols} grid")
        
        # Remove labels and borders
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {output_path}")
        
        # High definition version
        output_path_hd = output_path.replace('.png', '_HD.png')
        plt.savefig(output_path_hd, dpi=600, bbox_inches='tight',
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"  ✓ HD version saved: {output_path_hd}")
        
        plt.close()
    
    def generate_all_maps(self):
        """Generate all maps"""
        print("=" * 80)
        print("AMap Grid Generator")
        print("=" * 80)
        
        # Load data
        print("\nLoading data...")
        chengdu_data = self.load_chengdu_data()
        haikou_data = self.load_haikou_data()
        
        print(f"Chengdu data: {len(chengdu_data):,} records")
        print(f"Haikou data: {len(haikou_data):,} records")
        
        # Generate Chengdu map
        self.generate_city_map(
            'Chengdu',
            self.chengdu_bounds,
            chengdu_data,
            '/root/lanyun-tmp/data_code/chengdu_grid_map.png'
        )
        
        # Generate Haikou map
        self.generate_city_map(
            'Haikou',
            self.haikou_bounds,
            haikou_data,
            '/root/lanyun-tmp/data_code/haikou_grid_map.png'
        )
        
        print("\n" + "=" * 80)
        print("✅ All maps generated successfully!")
        print("=" * 80)
        
        print("\nUsage instructions:")
        print("  • Images have no title, no axis labels, suitable for paper figures")
        print("  • Grid lines are blue (#0066CC), data points are red (#DC143C)")
        print("  • Uses AMap as background, showing real geographic information")
        print("  • Boundaries based on dataset core area (90th percentile)")
        print("  • Suitable for academic use such as papers, presentations")


def main():
    """Main function"""
    generator = AMapGridGenerator()
    generator.generate_all_maps()


if __name__ == '__main__':
    main()
