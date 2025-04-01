import h5py
import os
import numpy as np
from .lc2latlon import *

class fy4_L1:
    NOMSatHeight = 42164    # 42164km指的是地心到卫星的距离，文件属性信息中的35786km指的是地表到卫星的距离
    NOMSubSatLon = None     # 星下点经度
    resolution = None       # 空间分辨率(km)
    data = {}
    vars = []
    def __init__(self,fpath):
        """
        Parameters
        -----
        fpath : str
            数据文件路径
        """
        # 读取hdf文件
        self.FileInfo = h5py.File(fpath, "r")
        # 获得文件属性信息
        attrs = dict(self.FileInfo.attrs)
        self.attrs = attrs

        if 'File Name' in attrs.keys():
            self.File_name = attrs['File Name'].decode('utf8')
        else:
            _, self.File_name = os.path.split(fpath)
        
        # 优先根据文件属性信息获取星下点经度, 没有相关信息则根据文件名获取(需确保文件名符合官方命名规则)
        if 'NOMSubSatLon' in attrs.keys():
            self.NOMSubSatLon = attrs['NOMSubSatLon']
        elif 'NOMCenterLon' in attrs.keys():
            self.NOMSubSatLon = attrs['NOMCenterLon']
        else:
            subsatlon_str = self.File_name.split('_')[-9]
            if len(subsatlon_str)!=5 or subsatlon_str[-1]!='E':
                raise RuntimeError("非标准文件名, 星下点经度信息缺失！")
            self.NOMSubSatLon = float(subsatlon_str[:-1])/10
        
        resolution_str = self.File_name.split('_')[-2]
        if resolution_str[-2:] in ['km', 'KM']:
            self.resolution = float(resolution_str[:-2])*1000
        elif resolution_str[-1:] in ['m', 'M']:
            self.resolution = float(resolution_str[:-1])
        else:
            raise RuntimeError("非标准文件名, 空间分辨率信息缺失！")
        
        # 获取数据的观测起止时间
        self.Observing_Beginning_DateTime = attrs['Observing Beginning Date'].decode('utf-8') + ' ' + attrs['Observing Beginning Time'].decode('utf-8')
        self.Observing_Ending_DateTime = attrs['Observing Ending Date'].decode('utf-8') + ' ' + attrs['Observing Ending Time'].decode('utf-8')

        # 获取标称行列号
        Begin_Line_Number = int(attrs['Begin Line Number'])    # 标称下的起始行号
        End_Line_Number = int(attrs['End Line Number'])        # 标称下的末尾行号
        Begin_Pixel_Number = int(attrs['Begin Pixel Number'])  # 标称下的起始列号
        End_Pixel_Number = int(attrs['End Pixel Number'])      # 标称下的末尾列号
        self.columns = np.arange(Begin_Pixel_Number, End_Pixel_Number+1, 1)
        self.lines = np.arange(Begin_Line_Number, End_Line_Number+1, 1)
    
    def __getitem__(self, varname):
        """
        允许以名称索引获取变量数值 
        """
        # 找到varname所在的group
        if varname in self.vars:
            return self.data[varname]
        elif varname in ['lat', 'lon']:
            return getattr(self, varname)
        else:
            raise ValueError(f"{varname} is not found!")

    def lc2latlon(self):
        """
        将行列号转为经纬度坐标
        """
        xx, yy = np.meshgrid(self.columns, self.lines)
        lon, lat = lc2latlon(xx.flatten(), yy.flatten(), sat='FY4', resolution=self.resolution, sub_lon=self.NOMSubSatLon)
        self.lon = lon.reshape(xx.shape)
        self.lat = lat.reshape(yy.shape)

class agri(fy4_L1):
    """
    读取风云4A-AGRI的L1数据
    """
    def __init__(self, fpath):
        super().__init__(fpath)
        self.read_data()
        self.lc2latlon()
    
    def read_data(self):
        # 获取各通道数值并进行辐射定标
        self.vars = [i for i in self.FileInfo.keys() if i[:10]=='NOMChannel']
        self.data = {v:self.FileInfo[v][:] for v in self.vars}
    
    def calibrate(self):
        """
        辐射定标
        """
        calibrated_data = {}
        for v in self.vars:
            nom = self.FileInfo[v]
            cal = self.FileInfo['CALChannel'+v[10:]]
            nom_channel = nom[:]
            cal_channel = cal[:]
        
            # 读取数据集属性
            nom_min, nom_max = nom.attrs['valid_range']
            cal_min, cal_max = cal.attrs['valid_range']
            nom_fill_value = nom.attrs['FillValue']
            cal_fill_value = cal.attrs['FillValue']

            # 数据集掩码和填充值预准备
            nom_mask = (nom_channel >= nom_min) & (nom_channel <= nom_max) & (nom_channel != nom_fill_value)
            cal_mask = (cal_channel >= cal_min) | (cal_channel <= cal_max)
            cal_channel[cal_channel == cal_fill_value] = int(cal_min - 10)

            # 辐射定标
            target_channel = np.zeros_like(nom_channel, dtype=np.float32)
            target_channel[nom_mask] = cal_channel[cal_mask][nom_channel[nom_mask]]
        
            # 无效值处理(包括不在范围及填充值)
            target_channel[~nom_mask] = np.nan
            target_channel[target_channel == int(cal_min - 10)] = np.nan
            calibrated_data[v] = target_channel
        return calibrated_data
    
class giirs(fy4_L1):
    """
    读取风云4A-GIIRS的L1数据
    """
    def __init__(self, fpath):
        super().__init__(fpath)
        self.read_data()

    def __getitem__(self, varname):
        return self.data[varname]
    
    def get_wavelength(self, var):
        return self.data[var]['wavelength']

    def read_data(self):
        """
        读取辐射数据
        """
        self.vars = ['VIS', 'NEdRLW', 'NEdRMW', 'RealLW', 'RealMW']
        self.data = {}
        
        # 读取可见光数据并定标
        self.data['VIS'] = {'data':self.FileInfo['ES_ContVIS'],
                            'dims':['line', 'column'],
                            'latitude':self.FileInfo['VIS_Latitude'][:],
                            'longitude':self.FileInfo['VIS_Longitude'][:],
                            'wavelength':'single',
                            }
        # 读取长波红外数据
        self.data['NEdRLW'] = {'data':self.FileInfo['ES_NEdRLW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['IRLW_Latitude'][:],
                            'longitude':self.FileInfo['IRLW_Longitude'][:],
                            'wavelength':self.FileInfo['IRLW_VaildWaveLength'][:],
                            }
        self.data['RealLW'] = {'data':self.FileInfo['ES_RealLW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['IRLW_Latitude'][:],
                            'longitude':self.FileInfo['IRLW_Longitude'][:],
                            'wavelength':self.FileInfo['IRLW_VaildWaveLength'][:],
                            }
        # 读取中波红外数据
        self.data['NEdRMW'] = {'data':self.FileInfo['ES_NEdRMW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['IRMW_Latitude'][:],
                            'longitude':self.FileInfo['IRMW_Longitude'][:],
                            'wavelength':self.FileInfo['IRMW_VaildWaveLength'][:],
                            }
        self.data['RealMW'] = {'data':self.FileInfo['ES_RealMW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['IRMW_Latitude'][:],
                            'longitude':self.FileInfo['IRMW_Longitude'][:],
                            'wavelength':self.FileInfo['IRMW_VaildWaveLength'][:],
                            }
    def calibrate(self):
        """
        对FY4A/GIIRS的可见光数据进行定标

        Parameters
        -----
        param hdf_path : hdf-object
            HDF5文件对象
        param nom_channel_name : str
            待定标的数据集名称
        param cal_channel_name : str
            用于定标的数据集名称
        sat : str, default='fy4a'
            卫星名称
            
        returns
        -----
        target_channel : array
            辐射定标后的数据
        """
        # 读取数据集
        nom = self.FileInfo['ES_ContVIS']
        cal = self.FileInfo['ES_CalSTableVIS']
        nom_channel = nom[:]
        cal_channel = cal[:]

        # 读取数据集属性
        nom_min, nom_max = nom.attrs['valid_range']
        cal_min, cal_max = cal.attrs['valid_range']
        nom_fill_value = nom.attrs['FillValue']
        cal_fill_value = cal.attrs['FillValue']

        # 数据集掩码和填充值预准备
        nom_mask = (nom_channel >= nom_min) & (nom_channel <= nom_max) & (nom_channel != nom_fill_value)

        # 辐射定标
        target_channel = np.zeros_like(nom_channel, dtype=np.float32)
        cal_mask = (cal_channel >= cal_min) | (cal_channel <= cal_max)
        cal_channel[cal_channel == cal_fill_value] = int(cal_min - 10)
        target_channel[nom_mask] = cal_channel[cal_mask][nom_channel[nom_mask]]
    
        # 无效值处理(包括不在范围及填充值)
        target_channel[~nom_mask] = np.nan
        target_channel[target_channel == int(cal_min - 10)] = np.nan
    
        return {'VIS': target_channel}

