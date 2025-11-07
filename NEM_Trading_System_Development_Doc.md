# NEM电力市场量化交易系统 - 技术开发文档

## 目录
1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [数据层实现](#3-数据层实现)
4. [特征工程模块](#4-特征工程模块)
5. [预测模型系统](#5-预测模型系统)
6. [交易策略引擎](#6-交易策略引擎)
7. [风险管理系统](#7-风险管理系统)
8. [回测框架](#8-回测框架)
9. [部署方案](#9-部署方案)
10. [监控与维护](#10-监控与维护)

---

## 1. 项目概述

### 1.1 项目目标
构建一个基于AEMO（澳大利亚能源市场运营商）NEM（国家电力市场）数据的自动化量化交易系统，实现电力市场的价格预测、套利交易和风险管理。

### 1.2 核心功能
- **实时数据采集**：从AEMO获取5分钟粒度的市场数据
- **价格预测**：使用机器学习模型预测电价走势和尖峰
- **自动交易**：执行均值回归、跨区域套利等多种策略
- **风险控制**：实时监控VaR、累积价格阈值等风险指标
- **性能分析**：提供完整的回测和实盘分析工具

### 1.3 技术栈
```yaml
编程语言:
  - Python 3.10+
  - SQL

数据存储:
  - PostgreSQL (时序数据)
  - Redis (缓存)
  - Parquet (历史数据)

框架和库:
  - FastAPI (API服务)
  - Pandas/NumPy (数据处理)
  - XGBoost/LightGBM (机器学习)
  - TensorFlow (深度学习)
  - Asyncio (异步处理)

监控部署:
  - Docker/Kubernetes
  - Grafana (监控)
  - Airflow (任务调度)
```

### 1.4 市场特性
- **结算周期**：5分钟现货价格，每天288个交易区间
- **价格范围**：-$1,000 到 $17,500/MWh
- **交易区域**：NSW, QLD, VIC, SA, TAS 五个定价区域
- **累积价格阈值（CPT）**：$1,398,000（7天累积）

---

## 2. 系统架构

### 2.1 整体架构图
```
┌─────────────────────────────────────────────────────────────┐
│                         用户界面层                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Dashboard │  │ 策略配置 │  │ 风险监控 │  │ 报告系统 │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         API服务层                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            FastAPI REST API Gateway                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         业务逻辑层                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │交易执行  │  │策略引擎  │  │风险管理  │  │绩效分析  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         计算引擎层                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │特征工程  │  │模型预测  │  │信号生成  │  │回测引擎  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         数据访问层                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │AEMO采集  │  │数据清洗  │  │特征存储  │  │缓存管理  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         存储层                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │PostgreSQL│  │  Redis   │  │ Parquet  │  │   S3     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构
```
nem_trading_system/
├── src/
│   ├── data/
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── aemo_collector.py
│   │   │   ├── realtime_stream.py
│   │   │   └── historical_loader.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── data_cleaner.py
│   │   │   └── data_validator.py
│   │   └── storage/
│   │       ├── __init__.py
│   │       ├── database.py
│   │       └── cache_manager.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── price_features.py
│   │   ├── demand_features.py
│   │   ├── temporal_features.py
│   │   └── cross_region_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── price_predictor.py
│   │   ├── spike_classifier.py
│   │   └── ensemble_model.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── mean_reversion.py
│   │   ├── spike_capture.py
│   │   ├── cross_region_arbitrage.py
│   │   └── base_strategy.py
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_calculator.py
│   │   ├── position_manager.py
│   │   └── hedging_engine.py
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py
│   │   ├── performance_metrics.py
│   │   └── optimization.py
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── order_manager.py
│   │   ├── execution_engine.py
│   │   └── portfolio.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   └── middleware/
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── helpers.py
├── tests/
├── configs/
│   ├── config.yaml
│   ├── strategies.yaml
│   └── models.yaml
├── scripts/
│   ├── setup_database.py
│   ├── download_historical.py
│   └── train_models.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 3. 数据层实现

### 3.1 AEMO数据采集器

```python
# src/data/collectors/aemo_collector.py

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from nemosis import dynamic_data_compiler, cache_compiler
import logging
from dataclasses import dataclass

@dataclass
class AEMOConfig:
    """AEMO数据配置"""
    cache_dir: str = '/data/aemo_cache'
    regions: List[str] = None
    tables: List[str] = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = ['NSW1', 'QLD1', 'VIC1', 'SA1', 'TAS1']
        if self.tables is None:
            self.tables = [
                'DISPATCHPRICE',
                'TRADINGPRICE', 
                'DISPATCHREGIONSUM',
                'DISPATCH_UNIT_SCADA',
                'BIDPEROFFER',
                'BIDDAYOFFER',
                'PREDISPATCHPRICE',
                'P5MIN_REGIONSOLUTION',
                'INTERCONNECTORRES'
            ]

class AEMODataCollector:
    """AEMO数据采集主类"""
    
    def __init__(self, config: AEMOConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def fetch_realtime_prices(self) -> pd.DataFrame:
        """异步获取实时价格数据"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            # 使用NEMOSIS获取最新数据
            data = dynamic_data_compiler(
                start_time=start_time.strftime('%Y/%m/%d %H:%M:%S'),
                end_time=end_time.strftime('%Y/%m/%d %H:%M:%S'),
                table='DISPATCHPRICE',
                raw_data_cache=self.config.cache_dir,
                filter_cols=['INTERVENTION'],
                filter_values=([0])  # 排除干预调度
            )
            
            # 数据清洗和验证
            data = self._clean_price_data(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取实时价格失败: {e}")
            raise
    
    def fetch_historical_data(
        self, 
        table: str,
        start_date: datetime,
        end_date: datetime,
        regions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """获取历史数据"""
        
        regions = regions or self.config.regions
        
        # 使用parquet格式优化存储
        cache_compiler(
            start_time=start_date.strftime('%Y/%m/%d %H:%M:%S'),
            end_time=end_date.strftime('%Y/%m/%d %H:%M:%S'),
            table=table,
            raw_data_cache=self.config.cache_dir,
            fformat='parquet'
        )
        
        # 加载并过滤数据
        data = dynamic_data_compiler(
            start_time=start_date.strftime('%Y/%m/%d %H:%M:%S'),
            end_time=end_date.strftime('%Y/%m/%d %H:%M:%S'),
            table=table,
            raw_data_cache=self.config.cache_dir,
            filter_cols=['REGIONID'] if 'REGION' in self._get_table_columns(table) else None,
            filter_values=(regions,) if regions else None
        )
        
        return data
    
    def _clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗价格数据"""
        # 移除异常值
        price_cap = 17500
        price_floor = -1000
        
        data['RRP'] = data['RRP'].clip(price_floor, price_cap)
        
        # 处理缺失值
        data['RRP'] = data['RRP'].fillna(method='ffill')
        
        # 时间戳对齐（AEMO时间戳在间隔末尾）
        data.index = pd.to_datetime(data['SETTLEMENTDATE']) - timedelta(minutes=5)
        
        return data
    
    def _get_table_columns(self, table: str) -> List[str]:
        """获取表的列名"""
        # 这里可以维护一个表结构映射
        table_columns = {
            'DISPATCHPRICE': ['SETTLEMENTDATE', 'REGIONID', 'RRP', 'INTERVENTION'],
            'DISPATCHREGIONSUM': ['SETTLEMENTDATE', 'REGIONID', 'TOTALDEMAND', 'AVAILABLEGENERATION'],
            # ... 其他表
        }
        return table_columns.get(table, [])

class RealTimeDataStream:
    """实时数据流处理"""
    
    def __init__(self, collector: AEMODataCollector):
        self.collector = collector
        self.is_running = False
        
    async def start_stream(self, callback):
        """启动实时数据流"""
        self.is_running = True
        
        while self.is_running:
            try:
                # 每5分钟获取一次数据
                data = await self.collector.fetch_realtime_prices()
                
                # 调用回调函数处理数据
                await callback(data)
                
                # 等待下一个5分钟间隔
                await asyncio.sleep(300)
                
            except Exception as e:
                logging.error(f"数据流错误: {e}")
                await asyncio.sleep(60)  # 错误后等待1分钟重试
    
    def stop_stream(self):
        """停止数据流"""
        self.is_running = False
```

### 3.2 数据存储管理

```python
# src/data/storage/database.py

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import pandas as pd
from typing import Optional, Dict, Any
import redis
import json

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            db=config['redis']['db']
        )
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = psycopg2.connect(
            host=self.config['postgres']['host'],
            port=self.config['postgres']['port'],
            database=self.config['postgres']['database'],
            user=self.config['postgres']['user'],
            password=self.config['postgres']['password']
        )
        try:
            yield conn
        finally:
            conn.close()
    
    def create_tables(self):
        """创建数据表"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # 价格数据表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS spot_prices (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        region VARCHAR(10) NOT NULL,
                        price DECIMAL(10, 2),
                        demand DECIMAL(10, 2),
                        available_generation DECIMAL(10, 2),
                        UNIQUE(timestamp, region)
                    );
                    CREATE INDEX idx_spot_prices_timestamp ON spot_prices(timestamp);
                    CREATE INDEX idx_spot_prices_region ON spot_prices(region);
                """)
                
                # 交易记录表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        strategy VARCHAR(50),
                        region VARCHAR(10),
                        action VARCHAR(10),
                        quantity DECIMAL(10, 2),
                        price DECIMAL(10, 2),
                        pnl DECIMAL(10, 2),
                        status VARCHAR(20)
                    );
                """)
                
                # 特征数据表
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS features (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        region VARCHAR(10) NOT NULL,
                        feature_name VARCHAR(100),
                        feature_value DECIMAL(20, 6),
                        UNIQUE(timestamp, region, feature_name)
                    );
                """)
                
                conn.commit()
    
    def insert_price_data(self, data: pd.DataFrame):
        """插入价格数据"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in data.iterrows():
                    cur.execute("""
                        INSERT INTO spot_prices 
                        (timestamp, region, price, demand, available_generation)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp, region) 
                        DO UPDATE SET 
                            price = EXCLUDED.price,
                            demand = EXCLUDED.demand,
                            available_generation = EXCLUDED.available_generation
                    """, (
                        row['timestamp'],
                        row['region'],
                        row['price'],
                        row.get('demand'),
                        row.get('available_generation')
                    ))
                conn.commit()
    
    def get_latest_prices(self, regions: Optional[List[str]] = None) -> pd.DataFrame:
        """获取最新价格"""
        query = """
            SELECT DISTINCT ON (region) 
                timestamp, region, price, demand, available_generation
            FROM spot_prices
            {}
            ORDER BY region, timestamp DESC
        """
        
        where_clause = ""
        if regions:
            where_clause = f"WHERE region IN ({','.join(['%s']*len(regions))})"
            
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query.format(where_clause),
                conn,
                params=regions if regions else None
            )
        return df
    
    def cache_features(self, key: str, features: Dict, ttl: int = 300):
        """缓存特征数据到Redis"""
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(features, default=str)
        )
    
    def get_cached_features(self, key: str) -> Optional[Dict]:
        """从Redis获取缓存的特征"""
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None
```

---

## 4. 特征工程模块

### 4.1 特征生成器

```python
# src/features/price_features.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """特征配置"""
    lookback_periods: List[int] = None
    volatility_windows: List[int] = None
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [1, 6, 12, 24, 48, 144, 288]  # 5min到24h
        if self.volatility_windows is None:
            self.volatility_windows = [12, 48, 144, 288]  # 1h到24h

class PriceFeatureGenerator:
    """价格特征生成器"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.price_cap = 17500
        self.price_floor = -1000
        self.cpt_threshold = 1398000
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成所有价格特征"""
        features = pd.DataFrame(index=data.index)
        
        # 基础价格特征
        features = pd.concat([features, self._price_momentum_features(data)], axis=1)
        features = pd.concat([features, self._price_volatility_features(data)], axis=1)
        features = pd.concat([features, self._price_distribution_features(data)], axis=1)
        
        # 累积价格特征
        features = pd.concat([features, self._cumulative_price_features(data)], axis=1)
        
        # 极值特征
        features = pd.concat([features, self._extreme_price_features(data)], axis=1)
        
        return features
    
    def _price_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """价格动量特征"""
        features = pd.DataFrame(index=data.index)
        
        for period in self.config.lookback_periods:
            # 价格变化
            features[f'price_change_{period}'] = data['RRP'].diff(period)
            
            # 价格收益率
            features[f'price_return_{period}'] = data['RRP'].pct_change(period)
            
            # 价格加速度
            if period > 1:
                features[f'price_acceleration_{period}'] = (
                    features[f'price_change_{period}'].diff(1)
                )
        
        return features
    
    def _price_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """价格波动率特征"""
        features = pd.DataFrame(index=data.index)
        
        for window in self.config.volatility_windows:
            # 标准差
            features[f'volatility_{window}'] = data['RRP'].rolling(window).std()
            
            # 波动率比率
            features[f'volatility_ratio_{window}'] = (
                features[f'volatility_{window}'] / 
                data['RRP'].rolling(window * 7).std()
            )
            
            # 实现波动率
            returns = data['RRP'].pct_change()
            features[f'realized_volatility_{window}'] = (
                returns.rolling(window).std() * np.sqrt(288 * 365)  # 年化
            )
            
            # GARCH特征（简化版）
            squared_returns = returns ** 2
            features[f'garch_volatility_{window}'] = (
                squared_returns.ewm(span=window).mean() ** 0.5
            )
        
        return features
    
    def _price_distribution_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """价格分布特征"""
        features = pd.DataFrame(index=data.index)
        
        for window in [48, 144, 288]:  # 4h, 12h, 24h
            # 分位数
            for quantile in [0.05, 0.25, 0.5, 0.75, 0.95]:
                features[f'price_quantile_{int(quantile*100)}_{window}'] = (
                    data['RRP'].rolling(window).quantile(quantile)
                )
            
            # 偏度和峰度
            features[f'price_skewness_{window}'] = data['RRP'].rolling(window).skew()
            features[f'price_kurtosis_{window}'] = data['RRP'].rolling(window).kurt()
            
            # Z-score
            mean = data['RRP'].rolling(window).mean()
            std = data['RRP'].rolling(window).std()
            features[f'price_zscore_{window}'] = (data['RRP'] - mean) / std
        
        return features
    
    def _cumulative_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """累积价格特征"""
        features = pd.DataFrame(index=data.index)
        
        # 7天累积价格（CPT相关）
        features['cumulative_price_7d'] = data['RRP'].rolling(2016).sum()
        features['cpt_utilization'] = features['cumulative_price_7d'] / self.cpt_threshold
        
        # CPT风险级别
        features['cpt_risk_level'] = pd.cut(
            features['cpt_utilization'],
            bins=[0, 0.5, 0.75, 0.9, 1.0, np.inf],
            labels=[1, 2, 3, 4, 5]  # 1=低风险, 5=超限
        ).astype(int)
        
        # 距离CPT的空间
        features['cpt_headroom'] = self.cpt_threshold - features['cumulative_price_7d']
        features['cpt_headroom_pct'] = features['cpt_headroom'] / self.cpt_threshold
        
        # 累积价格动量
        features['cumulative_price_momentum'] = features['cumulative_price_7d'].diff(12)
        
        return features
    
    def _extreme_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """极值价格特征"""
        features = pd.DataFrame(index=data.index)
        
        # 距离价格上下限
        features['dist_to_cap'] = (self.price_cap - data['RRP']) / self.price_cap
        features['dist_to_floor'] = (data['RRP'] - self.price_floor) / abs(self.price_floor)
        
        # 极值概率
        features['near_cap'] = (data['RRP'] > self.price_cap * 0.8).astype(int)
        features['near_floor'] = (data['RRP'] < 50).astype(int)
        
        # 尖峰检测
        rolling_mean = data['RRP'].rolling(288).mean()
        rolling_std = data['RRP'].rolling(288).std()
        features['is_spike'] = (
            np.abs(data['RRP'] - rolling_mean) > 3 * rolling_std
        ).astype(int)
        
        # 连续尖峰计数
        features['consecutive_spikes'] = (
            features['is_spike']
            .groupby((features['is_spike'] != features['is_spike'].shift()).cumsum())
            .cumsum()
        )
        
        return features
```

### 4.2 跨区域特征生成器

```python
# src/features/cross_region_features.py

import pandas as pd
import numpy as np
from typing import Dict, List

class CrossRegionFeatureGenerator:
    """跨区域特征生成器"""
    
    def __init__(self):
        self.regions = ['NSW1', 'VIC1', 'QLD1', 'SA1', 'TAS1']
        self.region_pairs = [
            ('NSW1', 'VIC1'),
            ('NSW1', 'QLD1'),
            ('VIC1', 'SA1'),
            ('SA1', 'TAS1')
        ]
    
    def generate_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """生成跨区域特征"""
        
        # 将各区域数据合并
        combined = pd.concat(
            {region: df for region, df in data.items()},
            axis=1
        )
        
        features = pd.DataFrame(index=combined.index)
        
        # 价差特征
        features = pd.concat([features, self._spread_features(combined)], axis=1)
        
        # 相关性特征
        features = pd.concat([features, self._correlation_features(combined)], axis=1)
        
        # 输电约束特征
        features = pd.concat([features, self._transmission_features(combined)], axis=1)
        
        return features
    
    def _spread_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """价差特征"""
        features = pd.DataFrame(index=data.index)
        
        for r1, r2 in self.region_pairs:
            spread_key = f'spread_{r1}_{r2}'
            
            # 基础价差
            features[spread_key] = data[r1]['RRP'] - data[r2]['RRP']
            
            # 价差统计
            for window in [48, 144, 288]:
                # 价差均值
                features[f'{spread_key}_mean_{window}'] = (
                    features[spread_key].rolling(window).mean()
                )
                
                # 价差标准差
                features[f'{spread_key}_std_{window}'] = (
                    features[spread_key].rolling(window).std()
                )
                
                # 价差Z-score
                features[f'{spread_key}_zscore_{window}'] = (
                    (features[spread_key] - features[f'{spread_key}_mean_{window}']) /
                    features[f'{spread_key}_std_{window}']
                )
                
            # 价差动量
            features[f'{spread_key}_momentum'] = features[spread_key].diff(6)
            
            # 价差反转信号
            features[f'{spread_key}_reversal'] = (
                np.abs(features[f'{spread_key}_zscore_288']) > 2
            ).astype(int)
        
        return features
    
    def _correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """相关性特征"""
        features = pd.DataFrame(index=data.index)
        
        # 滚动相关系数
        for window in [48, 144, 288]:
            for r1, r2 in self.region_pairs:
                corr_key = f'correlation_{r1}_{r2}_{window}'
                features[corr_key] = (
                    data[r1]['RRP'].rolling(window).corr(data[r2]['RRP'])
                )
                
                # 相关性变化
                features[f'{corr_key}_change'] = features[corr_key].diff(12)
        
        # 系统性风险指标（所有区域平均相关性）
        all_corr = []
        for window in [288]:
            corr_matrix = data[[r]['RRP'] for r in self.regions].rolling(window).corr()
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            features[f'systemic_correlation_{window}'] = avg_corr
        
        return features
    
    def _transmission_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """输电约束特征"""
        features = pd.DataFrame(index=data.index)
        
        # 这里需要实际的输电容量数据
        transmission_limits = {
            ('NSW1', 'VIC1'): 1350,
            ('NSW1', 'QLD1'): 1078,
            ('VIC1', 'SA1'): 650,
            ('SA1', 'TAS1'): 478
        }
        
        for (r1, r2), limit in transmission_limits.items():
            # 价差与输电限制的关系
            spread = data[r1]['RRP'] - data[r2]['RRP']
            
            # 输电压力指标
            features[f'transmission_stress_{r1}_{r2}'] = np.abs(spread) / 10  # 简化计算
            
            # 是否达到输电约束
            features[f'transmission_constrained_{r1}_{r2}'] = (
                features[f'transmission_stress_{r1}_{r2}'] > limit * 0.9
            ).astype(int)
        
        return features
```

---

## 5. 预测模型系统

### 5.1 价格预测模型

```python
# src/models/price_predictor.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

class PricePredictionSystem:
    """价格预测系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def build_models(self):
        """构建所有模型"""
        # XGBoost模型
        self.models['xgboost'] = self._build_xgboost()
        
        # LightGBM模型
        self.models['lightgbm'] = self._build_lightgbm()
        
        # LSTM模型
        self.models['lstm'] = self._build_lstm()
        
        # 集成模型
        self.models['ensemble'] = EnsembleModel(
            models=[self.models['xgboost'], self.models['lightgbm'], self.models['lstm']],
            weights=self.config.get('ensemble_weights', [0.4, 0.3, 0.3])
        )
    
    def _build_xgboost(self) -> xgb.XGBRegressor:
        """构建XGBoost模型"""
        return xgb.XGBRegressor(
            n_estimators=self.config.get('xgb_n_estimators', 1000),
            max_depth=self.config.get('xgb_max_depth', 7),
            learning_rate=self.config.get('xgb_learning_rate', 0.01),
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
    
    def _build_lightgbm(self) -> lgb.LGBMRegressor:
        """构建LightGBM模型"""
        return lgb.LGBMRegressor(
            n_estimators=self.config.get('lgb_n_estimators', 800),
            num_leaves=self.config.get('lgb_num_leaves', 31),
            max_depth=self.config.get('lgb_max_depth', -1),
            learning_rate=self.config.get('lgb_learning_rate', 0.05),
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_jobs=-1
        )
    
    def _build_lstm(self) -> tf.keras.Model:
        """构建LSTM模型"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """训练所有模型"""
        
        # 数据标准化
        self.scalers['features'] = StandardScaler()
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scalers['features'].transform(X_val)
        
        # 训练XGBoost
        self.models['xgboost'].fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)] if X_val is not None else None,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # 训练LightGBM
        self.models['lightgbm'].fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)] if X_val is not None else None,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # 训练LSTM（需要重塑数据）
        X_train_lstm = self._prepare_lstm_data(X_train_scaled)
        X_val_lstm = self._prepare_lstm_data(X_val_scaled) if X_val is not None else None
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        self.models['lstm'].fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # 获取特征重要性
        self._calculate_feature_importance(X_train)
    
    def predict(self, X: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """预测"""
        X_scaled = self.scalers['features'].transform(X)
        
        if model_name == 'lstm':
            X_scaled = self._prepare_lstm_data(X_scaled)
        
        return self.models[model_name].predict(X_scaled)
    
    def _prepare_lstm_data(self, X: np.ndarray, lookback: int = 12) -> np.ndarray:
        """准备LSTM数据格式"""
        # 简化版：假设X已经包含了时序特征
        return X.reshape((X.shape[0], 1, X.shape[1]))
    
    def _calculate_feature_importance(self, X_train: pd.DataFrame):
        """计算特征重要性"""
        # XGBoost特征重要性
        if 'xgboost' in self.models:
            self.feature_importance['xgboost'] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.models['xgboost'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        # LightGBM特征重要性
        if 'lightgbm' in self.models:
            self.feature_importance['lightgbm'] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.models['lightgbm'].feature_importances_
            }).sort_values('importance', ascending=False)
    
    def save_models(self, path: str):
        """保存模型"""
        joblib.dump(self.models['xgboost'], f"{path}/xgboost_model.pkl")
        joblib.dump(self.models['lightgbm'], f"{path}/lightgbm_model.pkl")
        self.models['lstm'].save(f"{path}/lstm_model.h5")
        joblib.dump(self.scalers, f"{path}/scalers.pkl")
    
    def load_models(self, path: str):
        """加载模型"""
        self.models['xgboost'] = joblib.load(f"{path}/xgboost_model.pkl")
        self.models['lightgbm'] = joblib.load(f"{path}/lightgbm_model.pkl")
        self.models['lstm'] = tf.keras.models.load_model(f"{path}/lstm_model.h5")
        self.scalers = joblib.load(f"{path}/scalers.pkl")

class EnsembleModel:
    """集成模型"""
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """加权平均预测"""
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred.flatten())
        
        weighted_pred = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_pred

class SpikePredictionModel:
    """价格尖峰预测模型"""
    
    def __init__(self, spike_threshold: float = 300):
        self.spike_threshold = spike_threshold
        self.classifier = None
        self.scaler = StandardScaler()
    
    def build_model(self):
        """构建尖峰分类器"""
        self.classifier = lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=31,
            max_depth=5,
            learning_rate=0.05,
            objective='binary',
            class_weight='balanced',
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def prepare_labels(self, prices: pd.Series) -> pd.Series:
        """准备尖峰标签"""
        # 定义尖峰：价格超过阈值或价格变化超过3倍标准差
        rolling_mean = prices.rolling(288).mean()
        rolling_std = prices.rolling(288).std()
        
        spike_labels = (
            (prices > self.spike_threshold) |
            (np.abs(prices - rolling_mean) > 3 * rolling_std)
        ).astype(int)
        
        return spike_labels
    
    def train(self, X: pd.DataFrame, prices: pd.Series):
        """训练尖峰预测模型"""
        # 准备标签
        y = self.prepare_labels(prices)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练分类器
        self.classifier.fit(X_scaled, y)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测尖峰概率"""
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)[:, 1]
```

---

## 6. 交易策略引擎

### 6.1 基础策略类

```python
# src/strategies/base_strategy.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

@dataclass
class Signal:
    """交易信号"""
    timestamp: pd.Timestamp
    strategy: str
    region: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    size: float
    metadata: Dict[str, Any] = None

class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.positions = {}
        self.signals_history = []
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """计算仓位大小"""
        pass
    
    def validate_signal(self, signal: Signal) -> bool:
        """验证信号有效性"""
        # 基础验证
        if signal.confidence < self.config.get('min_confidence', 0.5):
            return False
        
        if signal.size <= 0:
            return False
        
        # 检查是否有相反的未平仓位
        if signal.region in self.positions:
            current_position = self.positions[signal.region]
            if current_position['direction'] != signal.action:
                self.logger.warning(f"相反方向的未平仓位存在: {signal.region}")
        
        return True
    
    def update_positions(self, signal: Signal):
        """更新仓位"""
        if signal.action == 'HOLD':
            return
        
        self.positions[signal.region] = {
            'direction': signal.action,
            'size': signal.size,
            'entry_time': signal.timestamp,
            'entry_price': signal.metadata.get('price', 0)
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取策略表现指标"""
        if not self.signals_history:
            return {}
        
        df = pd.DataFrame(self.signals_history)
        
        return {
            'total_signals': len(df),
            'buy_signals': len(df[df['action'] == 'BUY']),
            'sell_signals': len(df[df['action'] == 'SELL']),
            'avg_confidence': df['confidence'].mean(),
            'signal_frequency': len(df) / ((df['timestamp'].max() - df['timestamp'].min()).days + 1)
        }
```

### 6.2 均值回归策略

```python
# src/strategies/mean_reversion.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base_strategy import BaseStrategy, Signal

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MeanReversion", config)
        self.ma_period = config.get('ma_period', 144)  # 12小时
        self.n_std = config.get('n_std', 2.5)
        self.min_spread = config.get('min_spread', 10)  # 最小价差
        self.max_position_size = config.get('max_position_size', 0.1)
    
    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> List[Signal]:
        """生成均值回归信号"""
        signals = []
        
        for region in data['region'].unique():
            region_data = data[data['region'] == region].copy()
            region_features = features[features.index.isin(region_data.index)]
            
            # 计算布林带
            ma = region_data['RRP'].rolling(self.ma_period).mean()
            std = region_data['RRP'].rolling(self.ma_period).std()
            upper_band = ma + self.n_std * std
            lower_band = ma - self.n_std * std
            
            # 当前价格
            current_price = region_data['RRP'].iloc[-1]
            current_time = region_data.index[-1]
            
            # 计算偏离程度
            z_score = (current_price - ma.iloc[-1]) / std.iloc[-1]
            
            # 生成信号
            signal = None
            if current_price < lower_band.iloc[-1] and (ma.iloc[-1] - current_price) > self.min_spread:
                # 价格低于下轨，买入信号
                signal = Signal(
                    timestamp=current_time,
                    strategy=self.name,
                    region=region,
                    action='BUY',
                    confidence=min(abs(z_score) / 3, 1.0),
                    size=self.calculate_position_size_internal(abs(z_score)),
                    metadata={
                        'price': current_price,
                        'ma': ma.iloc[-1],
                        'z_score': z_score,
                        'band': 'lower'
                    }
                )
            elif current_price > upper_band.iloc[-1] and (current_price - ma.iloc[-1]) > self.min_spread:
                # 价格高于上轨，卖出信号
                signal = Signal(
                    timestamp=current_time,
                    strategy=self.name,
                    region=region,
                    action='SELL',
                    confidence=min(z_score / 3, 1.0),
                    size=self.calculate_position_size_internal(z_score),
                    metadata={
                        'price': current_price,
                        'ma': ma.iloc[-1],
                        'z_score': z_score,
                        'band': 'upper'
                    }
                )
            
            # 检查止盈止损
            if region in self.positions:
                exit_signal = self.check_exit_conditions(region, current_price, ma.iloc[-1])
                if exit_signal:
                    signals.append(exit_signal)
            
            # 添加入场信号
            if signal and self.validate_signal(signal):
                signals.append(signal)
                self.update_positions(signal)
        
        return signals
    
    def calculate_position_size_internal(self, z_score: float) -> float:
        """基于Z-score计算仓位大小"""
        # Z-score越大，仓位越大，但不超过最大限制
        base_size = min(abs(z_score) / 4, 1.0)
        return base_size * self.max_position_size
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """计算实际仓位金额"""
        position_pct = signal.size
        return portfolio_value * position_pct
    
    def check_exit_conditions(self, region: str, current_price: float, ma: float) -> Optional[Signal]:
        """检查退出条件"""
        position = self.positions[region]
        
        # 均值回归目标达成
        if position['direction'] == 'BUY' and current_price >= ma:
            return Signal(
                timestamp=pd.Timestamp.now(),
                strategy=self.name,
                region=region,
                action='SELL',  # 平仓
                confidence=0.8,
                size=position['size'],
                metadata={'reason': 'target_reached', 'price': current_price}
            )
        elif position['direction'] == 'SELL' and current_price <= ma:
            return Signal(
                timestamp=pd.Timestamp.now(),
                strategy=self.name,
                region=region,
                action='BUY',  # 平仓
                confidence=0.8,
                size=position['size'],
                metadata={'reason': 'target_reached', 'price': current_price}
            )
        
        # 止损条件
        entry_price = position['entry_price']
        if position['direction'] == 'BUY':
            loss = (entry_price - current_price) / entry_price
            if loss > self.config.get('stop_loss', 0.05):  # 5%止损
                return Signal(
                    timestamp=pd.Timestamp.now(),
                    strategy=self.name,
                    region=region,
                    action='SELL',
                    confidence=1.0,
                    size=position['size'],
                    metadata={'reason': 'stop_loss', 'price': current_price, 'loss': loss}
                )
        
        return None
```

### 6.3 跨区域套利策略

```python
# src/strategies/cross_region_arbitrage.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from .base_strategy import BaseStrategy, Signal

class CrossRegionArbitrageStrategy(BaseStrategy):
    """跨区域套利策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("CrossRegionArbitrage", config)
        
        self.arbitrage_pairs = config.get('pairs', [
            ('NSW1', 'VIC1'),
            ('NSW1', 'QLD1'),
            ('VIC1', 'SA1')
        ])
        
        self.spread_thresholds = config.get('spread_thresholds', {
            ('NSW1', 'VIC1'): {'mean': 5, 'entry': 15, 'exit': 7},
            ('NSW1', 'QLD1'): {'mean': 8, 'entry': 20, 'exit': 10},
            ('VIC1', 'SA1'): {'mean': 10, 'entry': 25, 'exit': 12}
        })
        
        self.lookback_period = config.get('lookback_period', 288)  # 24小时
        self.min_zscore = config.get('min_zscore', 2.0)
        self.max_position_per_pair = config.get('max_position_per_pair', 0.05)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], features: pd.DataFrame) -> List[Signal]:
        """生成套利信号"""
        signals = []
        
        for r1, r2 in self.arbitrage_pairs:
            # 获取两个区域的数据
            data1 = data[r1]
            data2 = data[r2]
            
            # 确保时间对齐
            aligned_data = pd.DataFrame({
                f'{r1}_price': data1['RRP'],
                f'{r2}_price': data2['RRP']
            }).dropna()
            
            if len(aligned_data) < self.lookback_period:
                continue
            
            # 计算价差
            spread = aligned_data[f'{r1}_price'] - aligned_data[f'{r2}_price']
            
            # 价差统计
            spread_mean = spread.rolling(self.lookback_period).mean()
            spread_std = spread.rolling(self.lookback_period).std()
            z_score = (spread - spread_mean) / spread_std
            
            # 当前值
            current_spread = spread.iloc[-1]
            current_zscore = z_score.iloc[-1]
            current_time = aligned_data.index[-1]
            
            # 获取阈值
            thresholds = self.spread_thresholds.get((r1, r2), {})
            
            # 生成信号
            pair_key = f"{r1}_{r2}"
            
            if pair_key not in self.positions:
                # 开仓信号
                signal = self._generate_entry_signal(
                    r1, r2, current_spread, current_zscore, 
                    spread_mean.iloc[-1], thresholds, current_time
                )
                if signal:
                    signals.extend(signal)  # 套利需要两个信号
            else:
                # 平仓信号
                exit_signals = self._generate_exit_signal(
                    r1, r2, current_spread, current_zscore,
                    spread_mean.iloc[-1], thresholds, current_time
                )
                if exit_signals:
                    signals.extend(exit_signals)
        
        return signals
    
    def _generate_entry_signal(
        self, r1: str, r2: str, 
        spread: float, z_score: float, 
        mean_spread: float, thresholds: Dict,
        timestamp: pd.Timestamp
    ) -> Optional[List[Signal]]:
        """生成开仓信号"""
        
        if abs(z_score) < self.min_zscore:
            return None
        
        entry_threshold = thresholds.get('entry', 15)
        
        signals = []
        
        if z_score > self.min_zscore and spread > entry_threshold:
            # 价差过大，预期收敛
            # 卖R1，买R2
            confidence = min(z_score / 3, 1.0)
            
            signals.append(Signal(
                timestamp=timestamp,
                strategy=self.name,
                region=r1,
                action='SELL',
                confidence=confidence,
                size=self.max_position_per_pair,
                metadata={
                    'pair': f"{r1}_{r2}",
                    'spread': spread,
                    'z_score': z_score,
                    'type': 'arbitrage_leg1'
                }
            ))
            
            signals.append(Signal(
                timestamp=timestamp,
                strategy=self.name,
                region=r2,
                action='BUY',
                confidence=confidence,
                size=self.max_position_per_pair,
                metadata={
                    'pair': f"{r1}_{r2}",
                    'spread': spread,
                    'z_score': z_score,
                    'type': 'arbitrage_leg2'
                }
            ))
            
        elif z_score < -self.min_zscore and spread < -entry_threshold:
            # 价差过小，预期扩大
            # 买R1，卖R2
            confidence = min(abs(z_score) / 3, 1.0)
            
            signals.append(Signal(
                timestamp=timestamp,
                strategy=self.name,
                region=r1,
                action='BUY',
                confidence=confidence,
                size=self.max_position_per_pair,
                metadata={
                    'pair': f"{r1}_{r2}",
                    'spread': spread,
                    'z_score': z_score,
                    'type': 'arbitrage_leg1'
                }
            ))
            
            signals.append(Signal(
                timestamp=timestamp,
                strategy=self.name,
                region=r2,
                action='SELL',
                confidence=confidence,
                size=self.max_position_per_pair,
                metadata={
                    'pair': f"{r1}_{r2}",
                    'spread': spread,
                    'z_score': z_score,
                    'type': 'arbitrage_leg2'
                }
            ))
        
        if signals:
            # 记录套利对仓位
            self.positions[f"{r1}_{r2}"] = {
                'regions': (r1, r2),
                'entry_spread': spread,
                'entry_zscore': z_score,
                'entry_time': timestamp,
                'direction': 'convergence' if z_score > 0 else 'divergence'
            }
        
        return signals if signals else None
    
    def _generate_exit_signal(
        self, r1: str, r2: str,
        spread: float, z_score: float,
        mean_spread: float, thresholds: Dict,
        timestamp: pd.Timestamp
    ) -> Optional[List[Signal]]:
        """生成平仓信号"""
        
        pair_key = f"{r1}_{r2}"
        if pair_key not in self.positions:
            return None
        
        position = self.positions[pair_key]
        exit_threshold = thresholds.get('exit', 7)
        
        signals = []
        
        # 检查是否达到目标
        should_exit = False
        
        if position['direction'] == 'convergence':
            # 原来价差过大，现在检查是否已收敛
            if spread < mean_spread + exit_threshold or z_score < 0.5:
                should_exit = True
        else:  # divergence
            # 原来价差过小，现在检查是否已扩大
            if spread > mean_spread - exit_threshold or z_score > -0.5:
                should_exit = True
        
        # 检查止损
        entry_spread = position['entry_spread']
        if position['direction'] == 'convergence':
            if spread > entry_spread * 1.5:  # 价差继续扩大50%
                should_exit = True
        else:
            if spread < entry_spread * 1.5:  # 价差继续缩小50%
                should_exit = True
        
        if should_exit:
            # 平仓信号（反向操作）
            if position['direction'] == 'convergence':
                # 原来是卖R1买R2，现在买R1卖R2
                signals.append(Signal(
                    timestamp=timestamp,
                    strategy=self.name,
                    region=r1,
                    action='BUY',
                    confidence=0.9,
                    size=self.max_position_per_pair,
                    metadata={'pair': pair_key, 'type': 'close_arbitrage'}
                ))
                signals.append(Signal(
                    timestamp=timestamp,
                    strategy=self.name,
                    region=r2,
                    action='SELL',
                    confidence=0.9,
                    size=self.max_position_per_pair,
                    metadata={'pair': pair_key, 'type': 'close_arbitrage'}
                ))
            else:
                # 原来是买R1卖R2，现在卖R1买R2
                signals.append(Signal(
                    timestamp=timestamp,
                    strategy=self.name,
                    region=r1,
                    action='SELL',
                    confidence=0.9,
                    size=self.max_position_per_pair,
                    metadata={'pair': pair_key, 'type': 'close_arbitrage'}
                ))
                signals.append(Signal(
                    timestamp=timestamp,
                    strategy=self.name,
                    region=r2,
                    action='BUY',
                    confidence=0.9,
                    size=self.max_position_per_pair,
                    metadata={'pair': pair_key, 'type': 'close_arbitrage'}
                ))
            
            # 清除仓位记录
            del self.positions[pair_key]
        
        return signals if signals else None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """计算仓位大小"""
        return portfolio_value * signal.size
```

---

## 7. 风险管理系统

### 7.1 风险计算器

```python
# src/risk/risk_calculator.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import scipy.stats as stats

@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    cpt_utilization: float
    cpt_risk_level: str

class RiskCalculator:
    """风险计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cpt_threshold = 1398000  # 累积价格阈值
        self.risk_free_rate = config.get('risk_free_rate', 0.03)
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Any],
        market_data: pd.DataFrame,
        historical_data: pd.DataFrame
    ) -> RiskMetrics:
        """计算组合风险"""
        
        # 计算组合收益
        portfolio_returns = self._calculate_portfolio_returns(
            positions, historical_data
        )
        
        # VaR和CVaR
        var_95 = self._calculate_var(portfolio_returns, 0.95)
        var_99 = self._calculate_var(portfolio_returns, 0.99)
        cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
        cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # Sharpe和Sortino比率
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        
        # CPT风险
        cpt_utilization, cpt_risk_level = self._calculate_cpt_risk(
            positions, market_data
        )
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            cpt_utilization=cpt_utilization,
            cpt_risk_level=cpt_risk_level
        )
    
    def _calculate_portfolio_returns(
        self,
        positions: Dict[str, Any],
        historical_data: pd.DataFrame
    ) -> np.ndarray:
        """计算组合历史收益"""
        
        portfolio_values = []
        
        for date in historical_data.index:
            daily_value = 0
            for region, position in positions.items():
                if region in historical_data.columns:
                    price = historical_data.loc[date, region]
                    daily_value += position['size'] * price
            portfolio_values.append(daily_value)
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        return returns
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """计算VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """计算CVaR（条件VaR）"""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算Sharpe比率"""
        excess_returns = returns - self.risk_free_rate / 365
        return np.sqrt(365) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """计算Sortino比率"""
        excess_returns = returns - self.risk_free_rate / 365
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
        return np.sqrt(365) * excess_returns.mean() / downside_std
    
    def _calculate_cpt_risk(
        self,
        positions: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Tuple[float, str]:
        """计算CPT风险"""
        
        # 计算当前累积风险敞口
        cumulative_exposure = 0
        for region, position in positions.items():
            if region in market_data.index:
                current_price = market_data.loc[region, 'price']
                exposure = abs(position['size'] * current_price)
                cumulative_exposure += exposure
        
        # 计算利用率
        utilization = cumulative_exposure / self.cpt_threshold
        
        # 确定风险级别
        if utilization < 0.5:
            risk_level = 'LOW'
        elif utilization < 0.75:
            risk_level = 'MEDIUM'
        elif utilization < 0.9:
            risk_level = 'HIGH'
        elif utilization < 1.0:
            risk_level = 'CRITICAL'
        else:
            risk_level = 'EXCEEDED'
        
        return utilization, risk_level
    
    def stress_test(
        self,
        positions: Dict[str, Any],
        scenarios: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, float]:
        """压力测试"""
        
        if scenarios is None:
            scenarios = self._default_stress_scenarios()
        
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            scenario_loss = self._calculate_scenario_impact(
                positions, scenario_params
            )
            results[scenario_name] = scenario_loss
        
        return results
    
    def _default_stress_scenarios(self) -> Dict[str, Dict]:
        """默认压力测试场景"""
        return {
            'price_spike': {
                'price_change': 500,
                'probability': 0.05,
                'affected_regions': ['all']
            },
            'demand_surge': {
                'demand_multiplier': 1.3,
                'price_impact': 200,
                'probability': 0.1
            },
            'generation_failure': {
                'supply_reduction': 0.2,
                'price_impact': 1000,
                'probability': 0.02
            },
            'interconnector_failure': {
                'affected_pairs': [('NSW1', 'VIC1')],
                'spread_impact': 100,
                'probability': 0.03
            }
        }
    
    def _calculate_scenario_impact(
        self,
        positions: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> float:
        """计算场景影响"""
        
        total_impact = 0
        
        for region, position in positions.items():
            if 'price_change' in scenario:
                # 价格变化影响
                price_impact = scenario['price_change']
                if position['direction'] == 'BUY':
                    total_impact -= position['size'] * price_impact
                else:
                    total_impact += position['size'] * price_impact
            
            if 'price_impact' in scenario:
                # 直接价格影响
                total_impact -= abs(position['size']) * scenario['price_impact'] * 0.01
        
        return total_impact

class PositionManager:
    """仓位管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_position_per_region = config.get('max_position_per_region', 0.2)
        self.max_total_exposure = config.get('max_total_exposure', 0.8)
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.5)
    
    def validate_position(
        self,
        new_position: Dict[str, Any],
        existing_positions: Dict[str, Any],
        portfolio_value: float
    ) -> bool:
        """验证新仓位"""
        
        # 检查单个区域限制
        region = new_position['region']
        region_exposure = self._calculate_region_exposure(
            region, existing_positions, portfolio_value
        )
        
        new_exposure = abs(new_position['size'] * new_position.get('price', 100))
        if (region_exposure + new_exposure) / portfolio_value > self.max_position_per_region:
            return False
        
        # 检查总体暴露限制
        total_exposure = self._calculate_total_exposure(
            existing_positions, portfolio_value
        )
        if (total_exposure + new_exposure) / portfolio_value > self.max_total_exposure:
            return False
        
        # 检查相关性限制
        correlation_exposure = self._calculate_correlation_exposure(
            new_position, existing_positions, portfolio_value
        )
        if correlation_exposure > self.max_correlation_exposure:
            return False
        
        return True
    
    def _calculate_region_exposure(
        self,
        region: str,
        positions: Dict[str, Any],
        portfolio_value: float
    ) -> float:
        """计算区域暴露"""
        exposure = 0
        for pos_region, position in positions.items():
            if pos_region == region:
                exposure += abs(position['size'] * position.get('price', 100))
        return exposure
    
    def _calculate_total_exposure(
        self,
        positions: Dict[str, Any],
        portfolio_value: float
    ) -> float:
        """计算总暴露"""
        total = 0
        for position in positions.values():
            total += abs(position['size'] * position.get('price', 100))
        return total
    
    def _calculate_correlation_exposure(
        self,
        new_position: Dict[str, Any],
        existing_positions: Dict[str, Any],
        portfolio_value: float
    ) -> float:
        """计算相关性暴露"""
        # 简化版本：假设相邻区域相关性高
        high_corr_regions = {
            'NSW1': ['QLD1', 'VIC1'],
            'VIC1': ['NSW1', 'SA1', 'TAS1'],
            'QLD1': ['NSW1'],
            'SA1': ['VIC1'],
            'TAS1': ['VIC1']
        }
        
        correlated_exposure = 0
        new_region = new_position['region']
        
        for region, position in existing_positions.items():
            if region in high_corr_regions.get(new_region, []):
                correlated_exposure += abs(position['size'] * position.get('price', 100))
        
        return correlated_exposure / portfolio_value
```

---

## 8. 回测框架

### 8.1 回测引擎

```python
# src/backtesting/backtest_engine.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta

@dataclass
class Trade:
    """交易记录"""
    timestamp: pd.Timestamp
    strategy: str
    region: str
    action: str
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float = 0.0
    
@dataclass
class BacktestResult:
    """回测结果"""
    trades: List[Trade]
    equity_curve: pd.Series
    positions: pd.DataFrame
    metrics: Dict[str, float]
    daily_returns: pd.Series

class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0005,
        slippage_rate: float = 0.001
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(
        self,
        strategy: Any,
        data: pd.DataFrame,
        features: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """执行回测"""
        
        # 初始化
        self.cash = self.initial_capital
        self.positions = {}  # {region: {size, entry_price, entry_time}}
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        
        # 过滤数据
        mask = (data.index >= start_date) & (data.index <= end_date)
        backtest_data = data[mask].copy()
        backtest_features = features[mask].copy()
        
        # 主循环
        for timestamp in backtest_data.index.unique():
            try:
                # 获取当前时间的数据
                current_data = backtest_data.loc[timestamp]
                current_features = backtest_features.loc[timestamp]
                
                # 生成交易信号
                signals = strategy.generate_signals(current_data, current_features)
                
                # 执行交易
                for signal in signals:
                    self._execute_trade(signal, current_data)
                
                # 更新账户价值
                portfolio_value = self._calculate_portfolio_value(current_data)
                self.equity_curve.append(portfolio_value)
                self.timestamps.append(timestamp)
                
                # 风险检查
                if portfolio_value < self.initial_capital * 0.5:
                    self.logger.warning(f"账户价值低于50%，停止交易: {portfolio_value}")
                    break
                    
            except Exception as e:
                self.logger.error(f"回测错误 at {timestamp}: {e}")
                continue
        
        # 生成结果
        result = self._generate_result()
        return result
    
    def _execute_trade(self, signal: Any, market_data: pd.DataFrame):
        """执行交易"""
        
        region = signal.region
        action = signal.action
        
        # 获取当前价格
        if isinstance(market_data, pd.Series):
            current_price = market_data['RRP']
        else:
            region_data = market_data[market_data['region'] == region]
            if region_data.empty:
                return
            current_price = region_data['RRP'].iloc[0]
        
        # 计算滑点后的执行价格
        if action == 'BUY':
            execution_price = current_price * (1 + self.slippage_rate)
        else:
            execution_price = current_price * (1 - self.slippage_rate)
        
        # 计算交易数量
        position_value = self.cash * signal.size
        quantity = position_value / execution_price
        
        # 计算手续费
        commission = position_value * self.commission_rate
        
        # 执行交易
        if action == 'BUY':
            if self.cash >= position_value + commission:
                self.cash -= (position_value + commission)
                
                if region in self.positions:
                    # 加仓
                    old_pos = self.positions[region]
                    new_size = old_pos['size'] + quantity
                    new_avg_price = (
                        (old_pos['size'] * old_pos['entry_price'] + 
                         quantity * execution_price) / new_size
                    )
                    self.positions[region] = {
                        'size': new_size,
                        'entry_price': new_avg_price,
                        'entry_time': signal.timestamp
                    }
                else:
                    # 新仓
                    self.positions[region] = {
                        'size': quantity,
                        'entry_price': execution_price,
                        'entry_time': signal.timestamp
                    }
                
                # 记录交易
                trade = Trade(
                    timestamp=signal.timestamp,
                    strategy=signal.strategy,
                    region=region,
                    action=action,
                    quantity=quantity,
                    price=execution_price,
                    commission=commission,
                    slippage=execution_price - current_price
                )
                self.trades.append(trade)
                
        elif action == 'SELL':
            if region in self.positions:
                position = self.positions[region]
                
                # 计算卖出数量
                sell_quantity = min(quantity, position['size'])
                sell_value = sell_quantity * execution_price
                
                # 计算盈亏
                pnl = (execution_price - position['entry_price']) * sell_quantity
                
                # 更新现金
                self.cash += (sell_value - commission)
                
                # 更新仓位
                if sell_quantity >= position['size']:
                    # 清仓
                    del self.positions[region]
                else:
                    # 减仓
                    self.positions[region]['size'] -= sell_quantity
                
                # 记录交易
                trade = Trade(
                    timestamp=signal.timestamp,
                    strategy=signal.strategy,
                    region=region,
                    action=action,
                    quantity=sell_quantity,
                    price=execution_price,
                    commission=commission,
                    slippage=current_price - execution_price,
                    pnl=pnl
                )
                self.trades.append(trade)
    
    def _calculate_portfolio_value(self, market_data: pd.DataFrame) -> float:
        """计算组合价值"""
        
        portfolio_value = self.cash
        
        for region, position in self.positions.items():
            if isinstance(market_data, pd.Series):
                current_price = market_data['RRP']
            else:
                region_data = market_data[market_data['region'] == region]
                if not region_data.empty:
                    current_price = region_data['RRP'].iloc[0]
                else:
                    current_price = position['entry_price']  # 使用入场价格
            
            position_value = position['size'] * current_price
            portfolio_value += position_value
        
        return portfolio_value
    
    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        
        # 转换为Series
        equity_series = pd.Series(
            self.equity_curve,
            index=pd.to_datetime(self.timestamps)
        )
        
        # 计算日收益
        daily_returns = equity_series.pct_change().dropna()
        
        # 计算性能指标
        metrics = self._calculate_performance_metrics(
            equity_series, daily_returns, self.trades
        )
        
        # 仓位记录
        positions_df = pd.DataFrame(self.positions).T
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            positions=positions_df,
            metrics=metrics,
            daily_returns=daily_returns
        )
    
    def _calculate_performance_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """计算性能指标"""
        
        metrics = {}
        
        # 基础指标
        metrics['initial_capital'] = self.initial_capital
        metrics['final_capital'] = equity_curve.iloc[-1]
        metrics['total_return'] = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365
        metrics['annual_return'] = (
            (equity_curve.iloc[-1] / self.initial_capital) ** (1/years) - 1
        ) * 100 if years > 0 else 0
        
        # 风险指标
        metrics['volatility'] = daily_returns.std() * np.sqrt(365) * 100
        metrics['sharpe_ratio'] = (
            daily_returns.mean() / daily_returns.std() * np.sqrt(365)
        ) if daily_returns.std() > 0 else 0
        
        # Sortino比率
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0.001
        metrics['sortino_ratio'] = (
            daily_returns.mean() / downside_std * np.sqrt(365)
        ) if downside_std > 0 else 0
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min() * 100
        
        # 交易统计
        metrics['total_trades'] = len(trades)
        
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            metrics['win_rate'] = len(winning_trades) / len(trades) * 100
            
            if winning_trades:
                metrics['avg_win'] = np.mean([t.pnl for t in winning_trades])
            else:
                metrics['avg_win'] = 0
            
            if losing_trades:
                metrics['avg_loss'] = np.mean([t.pnl for t in losing_trades])
            else:
                metrics['avg_loss'] = 0
            
            # 盈亏比
            if metrics['avg_loss'] != 0:
                metrics['profit_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
            else:
                metrics['profit_loss_ratio'] = 0
            
            # 盈利因子
            total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
            total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
            
            # 总手续费
            metrics['total_commission'] = sum(t.commission for t in trades)
            metrics['total_slippage'] = sum(abs(t.slippage * t.quantity) for t in trades)
        
        return metrics
```

---

## 9. 部署方案

### 9.1 Docker配置

```dockerfile
# docker/Dockerfile

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 设置环境变量
ENV PYTHONPATH=/app
ENV TZ=Australia/Sydney

# 创建数据目录
RUN mkdir -p /data/aemo_cache /data/logs /data/models

# 运行应用
CMD ["python", "src/api/main.py"]
```

```yaml
# docker/docker-compose.yml

version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: nem_trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trader"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://trader:${DB_PASSWORD}@postgres:5432/nem_trading
      REDIS_URL: redis://redis:6379/0
      API_KEY: ${API_KEY}
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - ./logs:/app/logs
    restart: unless-stopped

  data_collector:
    build: .
    command: python scripts/data_collector.py
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://trader:${DB_PASSWORD}@postgres:5432/nem_trading
      REDIS_URL: redis://redis:6379/0
    volumes:
      - ./data:/data
    restart: unless-stopped

  scheduler:
    build: .
    command: airflow scheduler
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql://trader:${DB_PASSWORD}@postgres:5432/nem_trading
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    depends_on:
      - postgres

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

### 9.2 配置文件

```yaml
# configs/config.yaml

# 数据库配置
database:
  postgres:
    host: ${DB_HOST:localhost}
    port: 5432
    database: nem_trading
    user: trader
    password: ${DB_PASSWORD}
    pool_size: 10
    max_overflow: 20
  
  redis:
    host: ${REDIS_HOST:localhost}
    port: 6379
    db: 0
    decode_responses: true

# AEMO数据配置
aemo:
  cache_dir: /data/aemo_cache
  regions:
    - NSW1
    - QLD1
    - VIC1
    - SA1
    - TAS1
  tables:
    - DISPATCHPRICE
    - TRADINGPRICE
    - DISPATCHREGIONSUM
    - DISPATCH_UNIT_SCADA
    - BIDPEROFFER
    - PREDISPATCHPRICE
  update_interval: 300  # 5分钟

# 交易配置
trading:
  initial_capital: 1000000
  max_position_size: 0.1
  max_total_exposure: 0.8
  commission_rate: 0.0005
  slippage_rate: 0.001
  
  risk_limits:
    max_drawdown: 0.15
    var_limit: 50000
    cpt_safety_margin: 0.8
    
  strategies:
    mean_reversion:
      enabled: true
      weight: 0.4
      ma_period: 144
      n_std: 2.5
      min_spread: 10
      
    spike_capture:
      enabled: true
      weight: 0.3
      spike_threshold: 300
      confidence_threshold: 0.75
      
    cross_region_arbitrage:
      enabled: true
      weight: 0.3
      min_zscore: 2.0
      lookback_period: 288

# 模型配置
models:
  price_predictor:
    xgboost:
      n_estimators: 1000
      max_depth: 7
      learning_rate: 0.01
      
    lightgbm:
      n_estimators: 800
      num_leaves: 31
      learning_rate: 0.05
      
    ensemble_weights: [0.4, 0.3, 0.3]
  
  spike_classifier:
    threshold: 300
    lookback: 288
    
  update_frequency: daily

# API配置
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  cors_origins:
    - http://localhost:3000
    - http://localhost:8080
  
  rate_limiting:
    enabled: true
    default_limit: 100
    window_seconds: 60

# 日志配置
logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  file_handler:
    enabled: true
    filename: /app/logs/nem_trading.log
    max_bytes: 10485760  # 10MB
    backup_count: 5
  
  console_handler:
    enabled: true

# 监控配置
monitoring:
  metrics_port: 9090
  health_check_interval: 60
  
  alerts:
    email:
      enabled: false
      smtp_server: smtp.gmail.com
      smtp_port: 587
      sender: alerts@nemtrading.com
      recipients:
        - admin@nemtrading.com
    
    thresholds:
      max_drawdown: 0.1
      min_sharpe: 0.5
      cpt_utilization: 0.9

# 环境配置
environment: ${ENVIRONMENT:development}
```

---

## 10. 监控与维护

### 10.1 监控Dashboard

```python
# src/monitoring/dashboard.py

from flask import Flask, render_template, jsonify
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import json

app = Flask(__name__)

class DashboardService:
    """监控仪表板服务"""
    
    def __init__(self, db_manager, risk_calculator):
        self.db = db_manager
        self.risk_calc = risk_calculator
    
    def get_realtime_metrics(self) -> Dict:
        """获取实时指标"""
        
        # 获取最新价格
        latest_prices = self.db.get_latest_prices()
        
        # 获取当前仓位
        positions = self.db.get_current_positions()
        
        # 计算风险指标
        risk_metrics = self.risk_calc.calculate_portfolio_risk(
            positions, latest_prices, self.db.get_historical_data()
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'prices': latest_prices.to_dict('records'),
            'positions': positions,
            'risk': asdict(risk_metrics),
            'pnl': self._calculate_pnl(positions, latest_prices)
        }
    
    def get_performance_chart(self, period: str = '1d') -> str:
        """获取性能图表"""
        
        # 确定时间范围
        end_time = datetime.now()
        if period == '1d':
            start_time = end_time - timedelta(days=1)
        elif period == '1w':
            start_time = end_time - timedelta(weeks=1)
        elif period == '1m':
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=7)
        
        # 获取权益曲线
        equity_data = self.db.get_equity_curve(start_time, end_time)
        
        # 创建Plotly图表
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_data.index,
            y=equity_data['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Time',
            yaxis_title='Value ($)',
            hovermode='x unified'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def get_risk_heatmap(self) -> str:
        """获取风险热图"""
        
        regions = ['NSW1', 'QLD1', 'VIC1', 'SA1', 'TAS1']
        risk_matrix = []
        
        for region in regions:
            region_risks = []
            for metric in ['var_95', 'max_drawdown', 'volatility', 'cpt_risk']:
                value = self._calculate_region_risk(region, metric)
                region_risks.append(value)
            risk_matrix.append(region_risks)
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=['VaR 95%', 'Max DD', 'Volatility', 'CPT Risk'],
            y=regions,
            colorscale='RdYlGn_r'
        ))
        
        fig.update_layout(
            title='Risk Heatmap by Region',
            xaxis_title='Risk Metrics',
            yaxis_title='Regions'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def get_trade_analysis(self) -> Dict:
        """获取交易分析"""
        
        trades = self.db.get_recent_trades(days=7)
        
        if trades.empty:
            return {}
        
        analysis = {
            'total_trades': len(trades),
            'win_rate': (trades['pnl'] > 0).mean() * 100,
            'avg_pnl': trades['pnl'].mean(),
            'total_pnl': trades['pnl'].sum(),
            'by_strategy': trades.groupby('strategy')['pnl'].agg(['count', 'sum', 'mean']).to_dict(),
            'by_region': trades.groupby('region')['pnl'].sum().to_dict()
        }
        
        return analysis

# API路由
@app.route('/api/metrics')
def get_metrics():
    """获取实时指标API"""
    service = DashboardService(db_manager, risk_calculator)
    return jsonify(service.get_realtime_metrics())

@app.route('/api/performance/<period>')
def get_performance(period):
    """获取性能数据API"""
    service = DashboardService(db_manager, risk_calculator)
    return service.get_performance_chart(period)

@app.route('/api/risk/heatmap')
def get_risk_heatmap():
    """获取风险热图API"""
    service = DashboardService(db_manager, risk_calculator)
    return service.get_risk_heatmap()

@app.route('/api/trades/analysis')
def get_trade_analysis():
    """获取交易分析API"""
    service = DashboardService(db_manager, risk_calculator)
    return jsonify(service.get_trade_analysis())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 10.2 自动化任务调度

```python
# dags/nem_trading_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'nem_trader',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'nem_trading_pipeline',
    default_args=default_args,
    description='NEM Trading System Pipeline',
    schedule_interval='*/5 * * * *',  # 每5分钟运行
    catchup=False
)

# 任务定义
collect_data = PythonOperator(
    task_id='collect_realtime_data',
    python_callable=collect_aemo_data,
    dag=dag
)

generate_features = PythonOperator(
    task_id='generate_features',
    python_callable=calculate_features,
    dag=dag
)

predict_prices = PythonOperator(
    task_id='predict_prices',
    python_callable=run_predictions,
    dag=dag
)

generate_signals = PythonOperator(
    task_id='generate_signals',
    python_callable=generate_trading_signals,
    dag=dag
)

execute_trades = PythonOperator(
    task_id='execute_trades',
    python_callable=execute_trading_orders,
    dag=dag
)

update_risk = PythonOperator(
    task_id='update_risk_metrics',
    python_callable=update_risk_calculations,
    dag=dag
)

# 任务依赖
collect_data >> generate_features >> predict_prices >> generate_signals >> execute_trades >> update_risk

# 每日任务
daily_dag = DAG(
    'nem_daily_tasks',
    default_args=default_args,
    description='Daily maintenance tasks',
    schedule_interval='0 6 * * *',  # 每天早上6点
    catchup=False
)

model_retrain = PythonOperator(
    task_id='retrain_models',
    python_callable=retrain_prediction_models,
    dag=daily_dag
)

performance_report = PythonOperator(
    task_id='generate_performance_report',
    python_callable=generate_daily_report,
    dag=daily_dag
)

data_cleanup = BashOperator(
    task_id='cleanup_old_data',
    bash_command='find /data/aemo_cache -type f -mtime +30 -delete',
    dag=daily_dag
)

model_retrain >> performance_report >> data_cleanup
```

---

## 部署步骤

### 1. 环境准备
```bash
# 克隆代码
git clone https://github.com/your-repo/nem-trading-system.git
cd nem-trading-system

# 设置环境变量
cp .env.example .env
# 编辑.env文件，设置数据库密码、API密钥等

# 创建必要目录
mkdir -p data/aemo_cache logs data/models
```

### 2. 安装依赖
```bash
# Python依赖
pip install -r requirements.txt

# 或使用Docker
docker-compose build
```

### 3. 初始化数据库
```bash
# 运行数据库迁移
python scripts/setup_database.py

# 下载历史数据
python scripts/download_historical.py --start 2024-01-01 --end 2024-12-31
```

### 4. 训练模型
```bash
# 训练预测模型
python scripts/train_models.py --config configs/models.yaml
```

### 5. 启动服务
```bash
# 使用Docker Compose
docker-compose up -d

# 或手动启动
python src/api/main.py &
python scripts/data_collector.py &
airflow webserver -p 8080 &
airflow scheduler &
```

### 6. 访问系统
- API: http://localhost:8000
- Grafana监控: http://localhost:3000
- Airflow调度: http://localhost:8080

---

## 测试指南

### 运行单元测试
```bash
pytest tests/unit/ -v
```

### 运行集成测试
```bash
pytest tests/integration/ -v
```

### 运行回测验证
```bash
python scripts/run_backtest.py --strategy all --period 2024
```

---

## 维护建议

1. **每日检查**
   - 监控系统健康状态
   - 检查交易执行情况
   - 审查风险指标

2. **每周任务**
   - 分析策略表现
   - 优化模型参数
   - 清理过期数据

3. **每月任务**
   - 重新训练模型
   - 评估整体系统性能
   - 更新风险限制

---

## 联系方式

如有问题或需要支持，请联系：
- 技术支持：[support@nemtrading.com]
- 文档：[https://docs.nemtrading.com]

---

## 许可证

本项目使用 MIT 许可证。详见 LICENSE 文件。
