#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate grid partition map with real map background (Chengdu version)
For model diagram drawing material
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

# Set matplotlib parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False

class ChengduGridMapGenerator:
    """Generate Chengdu grid partition map"""
    
    def __init__(self, data_dir, output_path='chengdu_grid_map.png', basemap_path='chengdu.png'):
        self.data_dir = data_dir
        self.output_path = output_path
        self.basemap_path = basemap_path
        
        # Chengdu core urban area boundaries
        self.core_bounds = {
            'lon_min': 103.994821,
            'lon_max': 104.134852,
            'lat_min': 30.614351,
            'lat_max': 30.746338
        }
        
        # Grid parameters
        self.grid_rows = 20
        self.grid_cols = 20
    
    def load_sample_data(self, sample_ratio=0.1):
        """Load small sample data for visualization"""
        print("Loading sample data...")
        
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        # Only read the first file for speed
        df = pd.read_csv(csv_files[0])
        
        # Chengdu data column names are in Chinese
        lng_col = '上车位置经度'
        lat_col = '上车位置纬度'
        
        if lng_col not in df.columns:
            # Try English column names
            lng_col = 'starting_lng'
            lat_col = 'starting_lat'
        
        # Data cleaning
        df = df.dropna(subset=[lng_col, lat_col])
        
        # Sampling
        if sample_ratio < 1.0:
            df = df.sample(frac=sample_ratio, random_state=42)
        
        # Filter to core area
        core_df = df[
            (df[lng_col] >= self.core_bounds['lon_min']) &
            (df[lng_col] <= self.core_bounds['lon_max']) &
            (df[lat_col] >= self.core_bounds['lat_min']) &
            (df[lat_col] <= self.core_bounds['lat_max'])
        ].copy()
        
        # Standardize column names
        core_df['lng'] = core_df[lng_col]
        core_df['lat'] = core_df[lat_col]
        
        print(f"Sample data: {len(core_df):,} records")
        return core_df
    
    def create_grid_edges(self):
        """Create grid boundaries"""
        lon_edges = np.linspace(self.core_bounds['lon_min'], 
                               self.core_bounds['lon_max'], 
                               self.grid_cols + 1)
        lat_edges = np.linspace(self.core_bounds['lat_min'], 
                               self.core_bounds['lat_max'], 
                               self.grid_rows + 1)
        
        return lon_edges, lat_edges
    
    def generate_with_fallback(self, dpi=300, try_online=True):
        """Generate grid map with online map and local map fallback"""
        print("=" * 80)
        print("Chengdu Grid Partition Map Generator")
        print("=" * 80)
        
        # Load data
        print("\nStep 1: Loading sample data")
        sample_data = self.load_sample_data()
        
        print("\nStep 2: Creating grid")
        lon_edges, lat_edges = self.create_grid_edges()
        print(f"Grid configuration: {self.grid_rows}×{self.grid_cols} = {self.grid_rows*self.grid_cols} grids")
        
        print("\nStep 3: Generating map")
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Set axis limits (must be set before adding basemap)
        ax.set_xlim(self.core_bounds['lon_min'], self.core_bounds['lon_max'])
        ax.set_ylim(self.core_bounds['lat_min'], self.core_bounds['lat_max'])
        
        basemap_added = False
        
        # First try online maps
        if try_online:
            print("Attempting to load online map...")
            providers = [
                (ctx.providers.OpenStreetMap.Mapnik, "OpenStreetMap"),
                (ctx.providers.CartoDB.Positron, "CartoDB.Positron"),
                (ctx.providers.CartoDB.Voyager, "CartoDB.Voyager"),
            ]
            
            for provider, provider_name in providers:
                try:
                    print(f"  Trying {provider_name}...")
                    # Use default zoom, let contextily calculate automatically
                    ctx.add_basemap(ax, crs='EPSG:4326', source=provider, alpha=0.8)
                    print(f"  ✓ Successfully loaded {provider_name}")
                    basemap_added = True
                    break
                except Exception as e:
                    print(f"  ✗ {provider_name} failed: {str(e)[:100]}")
                    continue
        
        # If online maps fail, try local map
        if not basemap_added and os.path.exists(self.basemap_path):
            try:
                print("\nAttempting to load local map...")
                basemap_img = Image.open(self.basemap_path)
                ax.imshow(basemap_img, extent=[
                    self.core_bounds['lon_min'], self.core_bounds['lon_max'],
                    self.core_bounds['lat_min'], self.core_bounds['lat_max']
                ], alpha=0.9, aspect='auto', zorder=1)
                print(f"✓ Loaded local map: {self.basemap_path}")
                basemap_added = True
            except Exception as e:
                print(f"✗ Local map loading failed: {e}")
        
        # If all fail, use gradient background
        if not basemap_added:
            print("\nUsing default background")
            # Create gradient background
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            gradient = np.vstack((gradient, gradient))
            ax.imshow(gradient, extent=[self.core_bounds['lon_min'], self.core_bounds['lon_max'],
                                        self.core_bounds['lat_min'], self.core_bounds['lat_max']],
                      aspect='auto', cmap='Blues_r', alpha=0.1, zorder=0)
            ax.set_facecolor('#f8fbff')
        
        # Plot data points (red)
        if len(sample_data) > 0:
            ax.scatter(sample_data['lng'], sample_data['lat'],
                      s=8, alpha=0.7, c='#DC143C', edgecolors='white', 
                      linewidth=0.3, rasterized=True, zorder=5, label='Taxi Trips')
            print(f"✓ Plotted {len(sample_data):,} data points")
        
        # Draw grid lines (blue)
        for lon in lon_edges:
            ax.axvline(lon, color='#0066CC', linewidth=1.5, alpha=0.9, zorder=6)
        for lat in lat_edges:
            ax.axhline(lat, color='#0066CC', linewidth=1.5, alpha=0.9, zorder=6)
        print(f"✓ Drawn {self.grid_rows}×{self.grid_cols} grid")
        
        # Remove labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        # Remove borders
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save standard version
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"\n✓ Standard version saved: {self.output_path}")
        
        # High definition version
        output_path_hd = self.output_path.replace('.png', '_HD.png')
        plt.savefig(output_path_hd, dpi=600, bbox_inches='tight',
                   pad_inches=0, facecolor='white', edgecolor='none')
        print(f"✓ HD version saved: {output_path_hd}")
        
        plt.close()
        
        # Print statistics
        print("\n" + "=" * 80)
        print("📊 Map Information")
        print("=" * 80)
        print(f"Region: Chengdu core urban area")
        lon_span = (self.core_bounds['lon_max'] - self.core_bounds['lon_min']) * 92.0  # Chengdu latitude approx. 30°
        lat_span = (self.core_bounds['lat_max'] - self.core_bounds['lat_min']) * 111.0
        print(f"Coverage: {lon_span:.2f} km × {lat_span:.2f} km")
        print(f"Grid configuration: {self.grid_rows}×{self.grid_cols} = {self.grid_rows*self.grid_cols} grids")
        print(f"Each grid: {lon_span/self.grid_cols:.2f} km × {lat_span/self.grid_rows:.2f} km")
        print(f"Data points: {len(sample_data):,} records (sampled)")
        print("=" * 80)
        
        return self.output_path


def main():
    """Main function"""
    # Set paths
    data_dir = '/root/lanyun-tmp/data_code/raw_data/2016年11月成都网约车滴滴订单数据'
    output_path = '/root/lanyun-tmp/data_code/chengdu_grid_map.png'
    basemap_path = '/root/lanyun-tmp/data_code/chengdu.png'
    
    # Create generator
    generator = ChengduGridMapGenerator(data_dir, output_path, basemap_path)
    
    try:
        # Generate grid map - try online maps
        generator.generate_with_fallback(dpi=300, try_online=True)  # Use online maps
        
        print("\n✅ Generation completed!")
        print("\nOutput files:")
        print(f"  📄 Standard version (300 DPI): {output_path}")
        print(f"  📄 HD version (600 DPI): {output_path.replace('.png', '_HD.png')}")
        print("\nUsage instructions:")
        print("  • Images have no title, no axis labels, suitable for paper figures")
        print("  • Grid lines are blue (#0066CC), data points are red (#DC143C)")
        print("  • Uses online maps (CartoDB Positron) as background")
        print("  • Suitable for academic use such as papers, presentations")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
