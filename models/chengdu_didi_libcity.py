#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成都滴滴数据转换为LibCity格式 - 完整流程
方案B: 20×20网格，核心市区，对标TaxiBJ

参考：https://bigscity-libcity-docs.readthedocs.io/
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
        初始化转换器
        
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
            grid_rows: 网格行数
            grid_cols: 网格列数
            time_interval: 时间间隔（分钟）
            use_core_area: 是否只使用核心90%市区数据
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.time_interval = time_interval
        self.use_core_area = use_core_area
        self.dataset_name = f'ChengduDiDi{grid_rows}x{grid_cols}'
        
        # 确保输出目录存在
        util.ensure_dir(output_dir)
        
        print("=" * 80)
        print("成都滴滴数据 -> LibCity格式转换器")
        print("=" * 80)
        print(f"数据源: {data_dir}")
        print(f"输出目录: {output_dir}")
        print(f"网格配置: {grid_rows}×{grid_cols} = {grid_rows*grid_cols}个网格")
        print(f"时间间隔: {time_interval}分钟")
        print(f"使用核心区域: {use_core_area}")
        print("=" * 80)
        print()
    
    def step1_load_and_clean_data(self, max_files=None):
        """
        步骤1: 加载并清洗原始数据
        """
        print("【步骤1/4】加载并清洗数据...")
        print("-" * 80)
        
        # 获取所有CSV文件
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        if max_files:
            files = files[:max_files]
        
        print(f"找到 {len(files)} 个CSV文件")
        
        # 逐个加载
        data_list = []
        for file in tqdm(files, desc="加载文件"):
            try:
                file_path = os.path.join(self.data_dir, file)
                df = pd.read_csv(file_path)
                
                # 验证必要的列
                required_cols = ['订单ID', '开始计费时间', '结束计费时间', 
                               '上车位置经度', '上车位置纬度', '下车位置经度', '下车位置纬度']
                if all(col in df.columns for col in required_cols):
                    data_list.append(df)
            except Exception as e:
                print(f"警告: 加载文件 {file} 失败: {e}")
        
        # 合并所有数据
        print("合并数据...")
        raw_data = pd.concat(data_list, ignore_index=True)
        print(f"原始数据: {len(raw_data):,} 条记录")
        
        # 数据清洗
        print("\n数据清洗中...")
        
        # 1. 删除重复订单
        raw_data = raw_data.drop_duplicates(subset=['订单ID'])
        
        # 2. 删除空值
        raw_data = raw_data.dropna(subset=['订单ID', '开始计费时间', '结束计费时间',
                                           '上车位置经度', '上车位置纬度', 
                                           '下车位置经度', '下车位置纬度'])
        
        # 3. 时间格式转换
        raw_data['开始计费时间'] = pd.to_datetime(raw_data['开始计费时间'])
        raw_data['结束计费时间'] = pd.to_datetime(raw_data['结束计费时间'])
        
        # 4. 过滤时间异常
        raw_data = raw_data[raw_data['结束计费时间'] > raw_data['开始计费时间']]
        
        # 5. 计算行程时长（分钟）
        raw_data['行程时长'] = (raw_data['结束计费时间'] - raw_data['开始计费时间']).dt.total_seconds() / 60
        
        # 6. 过滤行程时长异常（2-120分钟）
        raw_data = raw_data[(raw_data['行程时长'] >= 2) & (raw_data['行程时长'] <= 120)]
        
        print(f"清洗后数据: {len(raw_data):,} 条记录")
        
        # 7. 筛选核心市区数据（可选）
        if self.use_core_area:
            print("\n筛选核心90%市区数据...")
            lon_coords = pd.concat([raw_data['上车位置经度'], raw_data['下车位置经度']])
            lat_coords = pd.concat([raw_data['上车位置纬度'], raw_data['下车位置纬度']])
            
            # 计算90%分位数范围
            lon_min = np.percentile(lon_coords, 5)
            lon_max = np.percentile(lon_coords, 95)
            lat_min = np.percentile(lat_coords, 5)
            lat_max = np.percentile(lat_coords, 95)
            
            # 筛选在核心区域内的订单
            raw_data = raw_data[
                (raw_data['上车位置经度'].between(lon_min, lon_max)) &
                (raw_data['上车位置纬度'].between(lat_min, lat_max)) &
                (raw_data['下车位置经度'].between(lon_min, lon_max)) &
                (raw_data['下车位置纬度'].between(lat_min, lat_max))
            ]
            
            print(f"核心区域数据: {len(raw_data):,} 条记录")
            print(f"经度范围: [{lon_min:.6f}, {lon_max:.6f}]")
            print(f"纬度范围: [{lat_min:.6f}, {lat_max:.6f}]")
        
        self.clean_data = raw_data
        print(f"\n✓ 步骤1完成，有效数据: {len(raw_data):,} 条")
        return raw_data
    
    def step2_create_grid_system(self):
        """
        步骤2: 创建网格系统
        """
        print("\n【步骤2/4】创建网格系统...")
        print("-" * 80)
        
        df = self.clean_data
        
        # 计算数据的经纬度范围
        lon_min = df[['上车位置经度', '下车位置经度']].min().min()
        lon_max = df[['上车位置经度', '下车位置经度']].max().max()
        lat_min = df[['上车位置纬度', '下车位置纬度']].min().min()
        lat_max = df[['上车位置纬度', '下车位置纬度']].max().max()
        
        # 创建网格边界
        lon_bins = np.linspace(lon_min, lon_max, self.grid_cols + 1)
        lat_bins = np.linspace(lat_min, lat_max, self.grid_rows + 1)
        
        print(f"网格配置: {self.grid_rows}行 × {self.grid_cols}列 = {self.grid_rows * self.grid_cols}个网格")
        
        # 估算每个网格的实际大小
        lon_per_grid = (lon_max - lon_min) / self.grid_cols
        lat_per_grid = (lat_max - lat_min) / self.grid_rows
        grid_width_km = lon_per_grid * 96.5  # 成都约北纬30度
        grid_height_km = lat_per_grid * 111.0
        print(f"单个网格大小: 约 {grid_width_km:.2f} km × {grid_height_km:.2f} km")
        
        # 为订单分配网格ID
        print("\n为订单分配网格ID...")
        df['pickup_grid_col'] = pd.cut(df['上车位置经度'], lon_bins, labels=False, include_lowest=True)
        df['pickup_grid_row'] = pd.cut(df['上车位置纬度'], lat_bins, labels=False, include_lowest=True)
        df['dropoff_grid_col'] = pd.cut(df['下车位置经度'], lon_bins, labels=False, include_lowest=True)
        df['dropoff_grid_row'] = pd.cut(df['下车位置纬度'], lat_bins, labels=False, include_lowest=True)
        
        # 计算网格ID（row * n_cols + col）
        df['pickup_grid_id'] = df['pickup_grid_row'] * self.grid_cols + df['pickup_grid_col']
        df['dropoff_grid_id'] = df['dropoff_grid_row'] * self.grid_cols + df['dropoff_grid_col']
        
        # 过滤分配失败的记录
        before = len(df)
        df = df.dropna(subset=['pickup_grid_id', 'dropoff_grid_id'])
        df['pickup_grid_id'] = df['pickup_grid_id'].astype(int)
        df['dropoff_grid_id'] = df['dropoff_grid_id'].astype(int)
        after = len(df)
        
        print(f"网格分配成功: {after:,}/{before:,} 条记录")
        
        # 保存网格信息
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
        print(f"\n✓ 步骤2完成")
        return df
    
    def step3_aggregate_flow(self):
        """
        步骤3: 时空流量聚合
        """
        print("\n【步骤3/4】时空流量聚合...")
        print("-" * 80)
        
        df = self.gridded_data
        
        # 时间处理
        df['date'] = df['开始计费时间'].dt.date
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        print(f"时间范围: {start_date} 到 {end_date}")
        
        # 创建时间窗口ID
        slots_per_day = 24 * 60 // self.time_interval
        print(f"每天时间窗口数: {slots_per_day}")
        
        # 计算全局时间索引
        def get_time_idx(row):
            days = (row['date'] - start_date).days
            minutes = row['开始计费时间'].hour * 60 + row['开始计费时间'].minute
            slot = minutes // self.time_interval
            return days * slots_per_day + slot
        
        print("计算时间索引...")
        df['time_idx'] = df.apply(get_time_idx, axis=1)
        
        max_time_idx = df['time_idx'].max()
        n_timesteps = max_time_idx + 1
        
        print(f"总时间步数: {n_timesteps}")
        
        # 统计inflow和outflow
        print("\n统计网格流量...")
        
        # Inflow: 以该网格为终点的出行次数（下车点数量）
        inflow = df.groupby(['time_idx', 'dropoff_grid_id']).size().reset_index(name='inflow')
        
        # Outflow: 以该网格为起点的出行次数（上车点数量）
        outflow = df.groupby(['time_idx', 'pickup_grid_id']).size().reset_index(name='outflow')
        
        # 创建完整的时空索引（确保所有时间和网格的组合都存在）
        print("生成完整时空矩阵...")
        time_indices = range(n_timesteps)
        grid_indices = range(self.grid_info['n_grids'])
        
        # 创建完整索引
        full_index = pd.MultiIndex.from_product(
            [time_indices, grid_indices],
            names=['time_idx', 'grid_id']
        )
        
        # 重新索引inflow（填充0）
        inflow_full = inflow.set_index(['time_idx', 'dropoff_grid_id'])['inflow'].reindex(
            full_index, fill_value=0
        ).reset_index()
        inflow_full.columns = ['time_idx', 'grid_id', 'inflow']
        
        # 重新索引outflow（填充0）
        outflow_full = outflow.set_index(['time_idx', 'pickup_grid_id'])['outflow'].reindex(
            full_index, fill_value=0
        ).reset_index()
        outflow_full.columns = ['time_idx', 'grid_id', 'outflow']
        
        # 合并
        flow_data = pd.merge(inflow_full, outflow_full, on=['time_idx', 'grid_id'])
        
        print(f"流量矩阵形状: {flow_data.shape}")
        print(f"平均inflow: {flow_data['inflow'].mean():.2f}")
        print(f"平均outflow: {flow_data['outflow'].mean():.2f}")
        
        # 计算零值率
        total_cells = len(flow_data)
        zero_cells = len(flow_data[(flow_data['inflow'] == 0) & (flow_data['outflow'] == 0)])
        zero_rate = zero_cells / total_cells * 100
        print(f"零值率: {zero_rate:.2f}%")
        
        self.flow_data = flow_data
        self.time_info = {
            'start_date': start_date,
            'end_date': end_date,
            'n_timesteps': n_timesteps,
            'slots_per_day': slots_per_day
        }
        
        print(f"\n✓ 步骤3完成")
        return flow_data
    
    def step4_convert_to_libcity(self):
        """
        步骤4: 转换为LibCity格式
        """
        print("\n【步骤4/4】转换为LibCity格式...")
        print("-" * 80)
        
        # 4.1 生成 .geo 文件
        print("\n生成 .geo 文件...")
        self._generate_geo_file()
        
        # 4.2 生成 .grid 文件
        print("生成 .grid 文件...")
        self._generate_grid_file()
        
        # 4.3 生成 config.json 文件
        print("生成 config.json 文件...")
        self._generate_config_file()
        
        print(f"\n✓ 步骤4完成")
        print("\n" + "=" * 80)
        print("转换完成！")
        print("=" * 80)
        print(f"输出文件位于: {self.output_dir}")
        print(f"  • {self.dataset_name}.geo")
        print(f"  • {self.dataset_name}.grid")
        print(f"  • config.json")
        print()
    
    def _generate_geo_file(self):
        """生成.geo文件（网格地理信息）"""
        geo_data = []
        
        lon_bins = self.grid_info['lon_bins']
        lat_bins = self.grid_info['lat_bins']
        
        for row_id in range(self.grid_rows):
            for col_id in range(self.grid_cols):
                geo_id = row_id * self.grid_cols + col_id
                
                # 构建多边形坐标（经纬度格式）
                lon_left = lon_bins[col_id]
                lon_right = lon_bins[col_id + 1]
                lat_bottom = lat_bins[row_id]
                lat_top = lat_bins[row_id + 1]
                
                # LibCity的Polygon格式: [[lon, lat], ...]
                coordinates = [[
                    [lon_left, lat_bottom],
                    [lon_right, lat_bottom],
                    [lon_right, lat_top],
                    [lon_left, lat_top],
                    [lon_left, lat_bottom]  # 闭合
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
        print(f"  保存: {geo_file} ({len(geo_df)} 个网格)")
    
    def _generate_grid_file(self):
        """生成.grid文件（时空流量数据）"""
        flow_data = self.flow_data.copy()
        
        # 转换时间索引为ISO格式时间
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
        
        print("  转换时间格式...")
        flow_data['time'] = flow_data['time_idx'].apply(time_idx_to_datetime)
        
        # 提取row_id和column_id
        flow_data['row_id'] = flow_data['grid_id'] // self.grid_cols
        flow_data['column_id'] = flow_data['grid_id'] % self.grid_cols
        
        # 构建LibCity .grid格式
        grid_data = flow_data[['time_idx', 'time', 'row_id', 'column_id', 'inflow', 'outflow']].copy()
        grid_data.insert(0, 'dyna_id', range(len(grid_data)))
        grid_data.insert(1, 'type', 'state')
        grid_data = grid_data[['dyna_id', 'type', 'time', 'row_id', 'column_id', 'inflow', 'outflow']]
        
        # 保存
        grid_file = os.path.join(self.output_dir, f'{self.dataset_name}.grid')
        grid_data.to_csv(grid_file, index=False)
        print(f"  保存: {grid_file} ({len(grid_data):,} 条记录)")
    
    def _generate_config_file(self):
        """生成config.json文件"""
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
                'time_intervals': self.time_interval * 60,  # 转换为秒
                'init_weight_inf_or_zero': 'inf',
                'set_weight_link_or_dist': 'dist',
                'calculate_weight_adj': False,
                'weight_adj_epsilon': 0.1
            }
        }
        
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"  保存: {config_file}")
    
    def generate_statistics_report(self):
        """生成数据集统计报告"""
        print("\n" + "=" * 80)
        print("数据集统计报告")
        print("=" * 80)
        
        flow_data = self.flow_data
        
        print(f"\n📊 基本信息")
        print(f"  数据集名称: {self.dataset_name}")
        print(f"  区域: 成都市核心区")
        print(f"  时间范围: {self.time_info['start_date']} ~ {self.time_info['end_date']}")
        
        n_days = (self.time_info['end_date'] - self.time_info['start_date']).days + 1
        print(f"  天数: {n_days} 天")
        
        print(f"\n🔢 数据规模")
        print(f"  网格配置: {self.grid_rows}×{self.grid_cols} = {self.grid_info['n_grids']} 个网格")
        print(f"  时间步数: {self.time_info['n_timesteps']}")
        print(f"  时间间隔: {self.time_interval} 分钟")
        print(f"  每日时间窗口: {self.time_info['slots_per_day']}")
        print(f"  总记录数: {len(flow_data):,}")
        
        print(f"\n📈 流量统计")
        print(f"  总trips: {int(flow_data['inflow'].sum()):,}")
        print(f"  平均inflow: {flow_data['inflow'].mean():.2f} 条/网格/时段")
        print(f"  平均outflow: {flow_data['outflow'].mean():.2f} 条/网格/时段")
        print(f"  最大inflow: {int(flow_data['inflow'].max())}")
        print(f"  最大outflow: {int(flow_data['outflow'].max())}")
        
        # 零值率
        zero_cells = len(flow_data[(flow_data['inflow'] == 0) & (flow_data['outflow'] == 0)])
        zero_rate = zero_cells / len(flow_data) * 100
        print(f"  零值率: {zero_rate:.2f}%")
        
        # 活跃网格
        active_grids = flow_data.groupby('grid_id')[['inflow', 'outflow']].sum()
        active_count = len(active_grids[(active_grids['inflow'] > 0) | (active_grids['outflow'] > 0)])
        print(f"  活跃网格数: {active_count}/{self.grid_info['n_grids']}")
        
        print(f"\n📏 网格信息")
        lon_span = (self.grid_info['lon_max'] - self.grid_info['lon_min']) * 96.5
        lat_span = (self.grid_info['lat_max'] - self.grid_info['lat_min']) * 111.0
        print(f"  覆盖范围: {lon_span:.2f} km × {lat_span:.2f} km")
        grid_w = lon_span / self.grid_cols
        grid_h = lat_span / self.grid_rows
        print(f"  单网格大小: {grid_w:.2f} km × {grid_h:.2f} km")
        
        print()


def main():
    """主函数"""
    # 配置参数
    converter = ChengduDiDiLibCityConverter(
        data_dir='data/2016年11月成都网约车滴滴订单数据',
        output_dir='output/ChengduDiDi20x20',
        grid_rows=20,
        grid_cols=20,
        time_interval=30,
        use_core_area=True
    )
    
    # 执行转换流程
    try:
        # 步骤1: 加载和清洗数据
        converter.step1_load_and_clean_data(max_files=None)  # None=使用所有文件
        
        # 步骤2: 创建网格系统
        converter.step2_create_grid_system()
        
        # 步骤3: 时空流量聚合
        converter.step3_aggregate_flow()
        
        # 步骤4: 转换为LibCity格式
        converter.step4_convert_to_libcity()
        
        # 生成统计报告
        converter.generate_statistics_report()
        
        print("✅ 转换成功！")
        print("\n下一步:")
        print("1. 将output目录中的文件复制到LibCity的数据目录")
        print("2. 在LibCity配置文件中指定数据集名称: ChengduDiDi20x20")
        print("3. 运行您的交通预测模型")
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

