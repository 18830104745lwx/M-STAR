#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chengdu DiDi Data Conversion to LibCity Format - Complete Process
Option B: 20×20 Grid, Core Urban Area, Aligned with TaxiBJ

Reference: https://bigscity-libcity-docs.readthedocs.io/
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import util

class ChengduDiDiLibCityConverter:
    def __init__(self, 
                 data_dir='data/2016年11月成都网约车滴滴订单数据',
                 output_dir='output/ChengduDiDi20x20',
                 grid_rows=20,
                 grid_cols=20,
                 time_interval=30,
                 use_core_area=True):
        """
        Initialize converter
        
        Args:
            data_dir: Raw data directory
            output_dir: Output directory
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns
            time_interval: Time interval (minutes)
            use_core_area: Whether to use only core 90% urban data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.time_interval = time_interval
        self.use_core_area = use_core_area
        self.dataset_name = f'ChengduDiDi{grid_rows}x{grid_cols}'
        
        # Ensure output directory exists
        util.ensure_dir(output_dir)
        
        print("=" * 80)
        print("Chengdu DiDi Data -> LibCity Format Converter")
        print("=" * 80)
        print(f"Data source: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Grid configuration: {grid_rows}×{grid_cols} = {grid_rows*grid_cols} grids")
        print(f"Time interval: {time_interval} minutes")
        print(f"Use core area: {use_core_area}")
        print("=" * 80)
        print()
    
    def step1_load_and_clean_data(self, max_files=None):
        """
        Step 1: Load and clean raw data
        """
        print("【Step 1/4】Loading and cleaning data...")
        print("-" * 80)
        
        # Get all CSV files
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        if max_files:
            files = files[:max_files]
        
        print(f"Found {len(files)} CSV files")
        
        # Load files one by one
        data_list = []
        for file in tqdm(files, desc="Loading files"):
            try:
                file_path = os.path.join(self.data_dir, file)
                df = pd.read_csv(file_path)
                
                # Verify required columns
                required_cols = ['OrderID', 'StartTime', 'EndTime', 
                               'PickupLon', 'PickupLat', 'DropoffLon', 'DropoffLat']
                if all(col in df.columns for col in required_cols):
                    data_list.append(df)
            except Exception as e:
                print(f"Warning: Failed to load file {file}: {e}")
        
        # Merge all data
        print("Merging data...")
        raw_data = pd.concat(data_list, ignore_index=True)
        print(f"Raw data: {len(raw_data):,} records")
        
        # Data cleaning
        print("\nData cleaning in progress...")
        
        # 1. Remove duplicate orders
        raw_data = raw_data.drop_duplicates(subset=['OrderID'])
        
        # 2. Remove null values
        raw_data = raw_data.dropna(subset=['OrderID', 'StartTime', 'EndTime',
                                           'PickupLon', 'PickupLat', 
                                           'DropoffLon', 'DropoffLat'])
        
        # 3. Convert time format
        raw_data['StartTime'] = pd.to_datetime(raw_data['StartTime'])
        raw_data['EndTime'] = pd.to_datetime(raw_data['EndTime'])
        
        # 4. Filter time anomalies
        raw_data = raw_data[raw_data['EndTime'] > raw_data['StartTime']]
        
        # 5. Calculate trip duration (minutes)
        raw_data['TripDuration'] = (raw_data['EndTime'] - raw_data['StartTime']).dt.total_seconds() / 60
        
        # 6. Filter trip duration anomalies (2-120 minutes)
        raw_data = raw_data[(raw_data['TripDuration'] >= 2) & (raw_data['TripDuration'] <= 120)]
        
        print(f"Cleaned data: {len(raw_data):,} records")
        
        # 7. Filter core urban area data (optional)
        if self.use_core_area:
            print("\nFiltering core 90% urban data...")
            lon_coords = pd.concat([raw_data['PickupLon'], raw_data['DropoffLon']])
            lat_coords = pd.concat([raw_data['PickupLat'], raw_data['DropoffLat']])
            
            # Calculate 90% percentile range
            lon_min = np.percentile(lon_coords, 5)
            lon_max = np.percentile(lon_coords, 95)
            lat_min = np.percentile(lat_coords, 5)
            lat_max = np.percentile(lat_coords, 95)
            
            # Filter orders within core area
            raw_data = raw_data[
                (raw_data['PickupLon'].between(lon_min, lon_max)) &
                (raw_data['PickupLat'].between(lat_min, lat_max)) &
                (raw_data['DropoffLon'].between(lon_min, lon_max)) &
                (raw_data['DropoffLat'].between(lat_min, lat_max))
            ]
            
            print(f"Core area data: {len(raw_data):,} records")
            print(f"Longitude range: [{lon_min:.6f}, {lon_max:.6f}]")
            print(f"Latitude range: [{lat_min:.6f}, {lat_max:.6f}]")
        
        self.clean_data = raw_data
        print(f"\n✓ Step 1 completed, valid data: {len(raw_data):,} records")
        return raw_data
    
    def step2_create_grid_system(self):
        """
        Step 2: Create grid system
        """
        print("\n【Step 2/4】Creating grid system...")
        print("-" * 80)
        
        df = self.clean_data
        
        # Calculate longitude/latitude range of data
        lon_min = df[['PickupLon', 'DropoffLon']].min().min()
        lon_max = df[['PickupLon', 'DropoffLon']].max().max()
        lat_min = df[['PickupLat', 'DropoffLat']].min().min()
        lat_max = df[['PickupLat', 'DropoffLat']].max().max()
        
        # Create grid boundaries
        lon_bins = np.linspace(lon_min, lon_max, self.grid_cols + 1)
        lat_bins = np.linspace(lat_min, lat_max, self.grid_rows + 1)
        
        print(f"Grid configuration: {self.grid_rows} rows × {self.grid_cols} columns = {self.grid_rows * self.grid_cols} grids")
        
        # Estimate actual size of each grid
        lon_per_grid = (lon_max - lon_min) / self.grid_cols
        lat_per_grid = (lat_max - lat_min) / self.grid_rows
        grid_width_km = lon_per_grid * 96.5  # Chengdu is approximately 30°N latitude
        grid_height_km = lat_per_grid * 111.0
        print(f"Single grid size: approx. {grid_width_km:.2f} km × {grid_height_km:.2f} km")
        
        # Assign grid IDs to orders
        print("\nAssigning grid IDs to orders...")
        df['pickup_grid_col'] = pd.cut(df['PickupLon'], lon_bins, labels=False, include_lowest=True)
        df['pickup_grid_row'] = pd.cut(df['PickupLat'], lat_bins, labels=False, include_lowest=True)
        df['dropoff_grid_col'] = pd.cut(df['DropoffLon'], lon_bins, labels=False, include_lowest=True)
        df['dropoff_grid_row'] = pd.cut(df['DropoffLat'], lat_bins, labels=False, include_lowest=True)
        
        # Calculate grid ID (row * n_cols + col)
        df['pickup_grid_id'] = df['pickup_grid_row'] * self.grid_cols + df['pickup_grid_col']
        df['dropoff_grid_id'] = df['dropoff_grid_row'] * self.grid_cols + df['dropoff_grid_col']
        
        # Filter records that failed grid assignment
        before = len(df)
        df = df.dropna(subset=['pickup_grid_id', 'dropoff_grid_id'])
        df['pickup_grid_id'] = df['pickup_grid_id'].astype(int)
        df['dropoff_grid_id'] = df['dropoff_grid_id'].astype(int)
        after = len(df)
        
        print(f"Grid assignment successful: {after:,}/{before:,} records")
        
        # Save grid information
        self.grid_info = {
            'lon_bins': lon_bins,
            'lat_bins': lat_bins,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'n_grids': self.grid_rows * self.grid_cols
        }
        
        self.gridded_data = df
        print(f"\n✓ Step 2 completed")
        return df
    
    def step3_aggregate_flow(self):
        """
        Step 3: Spatiotemporal flow aggregation
        """
        print("\n【Step 3/4】Spatiotemporal flow aggregation...")
        print("-" * 80)
        
        df = self.gridded_data
        
        # Time processing
        df['date'] = df['StartTime'].dt.date
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        print(f"Time range: {start_date} to {end_date}")
        
        # Create time window IDs
        slots_per_day = 24 * 60 // self.time_interval
        print(f"Time slots per day: {slots_per_day}")
        
        # Calculate global time index
        def get_time_idx(row):
            days = (row['date'] - start_date).days
            minutes = row['StartTime'].hour * 60 + row['StartTime'].minute
            slot = minutes // self.time_interval
            return days * slots_per_day + slot
        
        print("Calculating time indices...")
        df['time_idx'] = df.apply(get_time_idx, axis=1)
        
        max_time_idx = df['time_idx'].max()
        n_timesteps = max_time_idx + 1
        
        print(f"Total time steps: {n_timesteps}")
        
        # Calculate inflow and outflow
        print("\nCalculating grid flow...")
        
        # Inflow: Number of trips ending at this grid (dropoff count)
        inflow = df.groupby(['time_idx', 'dropoff_grid_id']).size().reset_index(name='inflow')
        
        # Outflow: Number of trips starting at this grid (pickup count)
        outflow = df.groupby(['time_idx', 'pickup_grid_id']).size().reset_index(name='outflow')
        
        # Create complete spatiotemporal index (ensure all time-grid combinations exist)
        print("Generating complete spatiotemporal matrix...")
        time_indices = range(n_timesteps)
        grid_indices = range(self.grid_info['n_grids'])
        
        # Create complete index
        full_index = pd.MultiIndex.from_product(
            [time_indices, grid_indices],
            names=['time_idx', 'grid_id']
        )
        
        # Reindex inflow (fill with 0)
        inflow_full = inflow.set_index(['time_idx', 'dropoff_grid_id'])['inflow'].reindex(
            full_index, fill_value=0
        ).reset_index()
        inflow_full.columns = ['time_idx', 'grid_id', 'inflow']
        
        # Reindex outflow (fill with 0)
        outflow_full = outflow.set_index(['time_idx', 'pickup_grid_id'])['outflow'].reindex(
            full_index, fill_value=0
        ).reset_index()
        outflow_full.columns = ['time_idx', 'grid_id', 'outflow']
        
        # Merge
        flow_data = pd.merge(inflow_full, outflow_full, on=['time_idx', 'grid_id'])
        
        print(f"Flow matrix shape: {flow_data.shape}")
        print(f"Average inflow: {flow_data['inflow'].mean():.2f}")
        print(f"Average outflow: {flow_data['outflow'].mean():.2f}")
        
        # Calculate zero-value rate
        total_cells = len(flow_data)
        zero_cells = len(flow_data[(flow_data['inflow'] == 0) & (flow_data['outflow'] == 0)])
        zero_rate = zero_cells / total_cells * 100
        print(f"Zero-value rate: {zero_rate:.2f}%")
        
        self.flow_data = flow_data
        self.time_info = {
            'start_date': start_date,
            'end_date': end_date,
            'n_timesteps': n_timesteps,
            'slots_per_day': slots_per_day
        }
        
        print(f"\n✓ Step 3 completed")
        return flow_data
    
    def step4_convert_to_libcity(self):
        """
        Step 4: Convert to LibCity format
        """
        print("\n【Step 4/4】Converting to LibCity format...")
        print("-" * 80)
        
        # 4.1 Generate .geo file
        print("\nGenerating .geo file...")
        self._generate_geo_file()
        
        # 4.2 Generate .grid file
        print("Generating .grid file...")
        self._generate_grid_file()
        
        # 4.3 Generate config.json file
        print("Generating config.json file...")
        self._generate_config_file()
        
        print(f"\n✓ Step 4 completed")
        print("\n" + "=" * 80)
        print("Conversion completed!")
        print("=" * 80)
        print(f"Output files are located at: {self.output_dir}")
        print(f"  • {self.dataset_name}.geo")
        print(f"  • {self.dataset_name}.grid")
        print(f"  • config.json")
        print()
    
    def _generate_geo_file(self):
        """Generate .geo file (grid geographic information)"""
        geo_data = []
        
        lon_bins = self.grid_info['lon_bins']
        lat_bins = self.grid_info['lat_bins']
        
        for row_id in range(self.grid_rows):
            for col_id in range(self.grid_cols):
                geo_id = row_id * self.grid_cols + col_id
                
                # Build polygon coordinates (longitude/latitude format)
                lon_left = lon_bins[col_id]
                lon_right = lon_bins[col_id + 1]
                lat_bottom = lat_bins[row_id]
                lat_top = lat_bins[row_id + 1]
                
                # LibCity Polygon format: [[lon, lat], ...]
                coordinates = [[
                    [lon_left, lat_bottom],
                    [lon_right, lat_bottom],
                    [lon_right, lat_top],
                    [lon_left, lat_top],
                    [lon_left, lat_bottom]  # Close the polygon
                ]]
                
                geo_data.append({
                    'geo_id': geo_id,
                    'type': 'Polygon',
                    'coordinates': str(coordinates),
                    'row_id': row_id,
                    'column_id': col_id
                })
        
        geo_df = pd.DataFrame(geo_data)
        geo_file = os.path.join(self.output_dir, f'{self.dataset_name}.geo')
        geo_df.to_csv(geo_file, index=False)
        print(f"  Saved: {geo_file} ({len(geo_df)} grids)")
    
    def _generate_grid_file(self):
        """Generate .grid file (spatiotemporal flow data)"""
        flow_data = self.flow_data.copy()
        
        # Convert time index to ISO format
        start_date = self.time_info['start_date']
        slots_per_day = self.time_info['slots_per_day']
        
        def time_idx_to_datetime(time_idx):
            days = time_idx // slots_per_day
            slot_in_day = time_idx % slots_per_day
            hours = (slot_in_day * self.time_interval) // 60
            minutes = (slot_in_day * self.time_interval) % 60
            
            dt = datetime.combine(start_date, datetime.min.time())
            dt = dt.replace(hour=hours, minute=minutes)
            dt = dt + pd.Timedelta(days=days)
            
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        print("  Converting time format...")
        flow_data['time'] = flow_data['time_idx'].apply(time_idx_to_datetime)
        
        # Extract row_id and column_id
        flow_data['row_id'] = flow_data['grid_id'] // self.grid_cols
        flow_data['column_id'] = flow_data['grid_id'] % self.grid_cols
        
        # Build LibCity .grid format
        grid_data = flow_data[['time_idx', 'time', 'row_id', 'column_id', 'inflow', 'outflow']].copy()
        grid_data.insert(0, 'dyna_id', range(len(grid_data)))
        grid_data.insert(1, 'type', 'state')
        grid_data = grid_data[['dyna_id', 'type', 'time', 'row_id', 'column_id', 'inflow', 'outflow']]
        
        # Save
        grid_file = os.path.join(self.output_dir, f'{self.dataset_name}.grid')
        grid_data.to_csv(grid_file, index=False)
        print(f"  Saved: {grid_file} ({len(grid_data):,} records)")
    
    def _generate_config_file(self):
        """Generate config.json file"""
        config = {
            'geo': {
                'including_types': ['Polygon'],
                'Polygon': {
                    'row_id': 'num',
                    'column_id': 'num'
                }
            },
            'grid': {
                'including_types': ['state'],
                'state': {
                    'row_id': self.grid_rows,
                    'column_id': self.grid_cols,
                    'inflow': 'num',
                    'outflow': 'num'
                }
            },
            'info': {
                'data_col': ['inflow', 'outflow'],
                'data_files': [self.dataset_name],
                'geo_file': self.dataset_name,
                'output_dim': 2,
                'time_intervals': self.time_interval * 60,  # Convert to seconds
                'init_weight_inf_or_zero': 'inf',
                'set_weight_link_or_dist': 'dist',
                'calculate_weight_adj': False,
                'weight_adj_epsilon': 0.1
            }
        }
        
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {config_file}")
    
    def generate_statistics_report(self):
        """Generate dataset statistics report"""
        print("\n" + "=" * 80)
        print("Dataset Statistics Report")
        print("=" * 80)
        
        flow_data = self.flow_data
        
        print(f"\n📊 Basic Information")
        print(f"  Dataset name: {self.dataset_name}")
        print(f"  Region: Chengdu core urban area")
        print(f"  Time range: {self.time_info['start_date']} ~ {self.time_info['end_date']}")
        
        n_days = (self.time_info['end_date'] - self.time_info['start_date']).days + 1
        print(f"  Number of days: {n_days} days")
        
        print(f"\n🔢 Data Scale")
        print(f"  Grid configuration: {self.grid_rows}×{self.grid_cols} = {self.grid_info['n_grids']} grids")
        print(f"  Time steps: {self.time_info['n_timesteps']}")
        print(f"  Time interval: {self.time_interval} minutes")
        print(f"  Daily time windows: {self.time_info['slots_per_day']}")
        print(f"  Total records: {len(flow_data):,}")
        
        print(f"\n📈 Flow Statistics")
        print(f"  Total trips: {int(flow_data['inflow'].sum()):,}")
        print(f"  Average inflow: {flow_data['inflow'].mean():.2f} trips/grid/time-slot")
        print(f"  Average outflow: {flow_data['outflow'].mean():.2f} trips/grid/time-slot")
        print(f"  Maximum inflow: {int(flow_data['inflow'].max())}")
        print(f"  Maximum outflow: {int(flow_data['outflow'].max())}")
        
        # Zero-value rate
        zero_cells = len(flow_data[(flow_data['inflow'] == 0) & (flow_data['outflow'] == 0)])
        zero_rate = zero_cells / len(flow_data) * 100
        print(f"  Zero-value rate: {zero_rate:.2f}%")
        
        # Active grids
        active_grids = flow_data.groupby('grid_id')[['inflow', 'outflow']].sum()
        active_count = len(active_grids[(active_grids['inflow'] > 0) | (active_grids['outflow'] > 0)])
        print(f"  Active grids: {active_count}/{self.grid_info['n_grids']}")
        
        print(f"\n📏 Grid Information")
        lon_span = (self.grid_info['lon_max'] - self.grid_info['lon_min']) * 96.5
        lat_span = (self.grid_info['lat_max'] - self.grid_info['lat_min']) * 111.0
        print(f"  Coverage area: {lon_span:.2f} km × {lat_span:.2f} km")
        grid_w = lon_span / self.grid_cols
        grid_h = lat_span / self.grid_rows
        print(f"  Single grid size: {grid_w:.2f} km × {grid_h:.2f} km")
        
        print()


def main():
    """Main function"""
    # Configuration parameters
    converter = ChengduDiDiLibCityConverter(
        data_dir='data/2016年11月成都网约车滴滴订单数据',
        output_dir='output/ChengduDiDi20x20',
        grid_rows=20,
        grid_cols=20,
        time_interval=30,
        use_core_area=True
    )
    
    # Execute conversion process
    try:
        # Step 1: Load and clean data
        converter.step1_load_and_clean_data(max_files=None)  # None=use all files
        
        # Step 2: Create grid system
        converter.step2_create_grid_system()
        
        # Step 3: Spatiotemporal flow aggregation
        converter.step3_aggregate_flow()
        
        # Step 4: Convert to LibCity format
        converter.step4_convert_to_libcity()
        
        # Generate statistics report
        converter.generate_statistics_report()
        
        print(" Conversion successful!")
        print("\nNext steps:")
        print("1. Copy the files in the output directory to LibCity's data directory")
        print("2. Specify dataset name in LibCity config: ChengduDiDi20x20")
        print("3. Run your traffic prediction model")
        
    except Exception as e:
        print(f"\n Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
