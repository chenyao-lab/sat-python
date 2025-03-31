import numpy as np
from FY4A import fy4_L1

class fy4b_agri(fy4_L1):
    """
    读取风云4B-AGRI/GHI的L1数据
    """
    def __init__(self, fpath):
        super().__init__(fpath)
        self.read_data()
        self.lc2latlon()
    
    def read_data(self):
        # 获取各通道数值并进行辐射定标
        self.vars = [i for i in self.FileInfo['Data'].keys() if i.startswith('NOMChannel')]
        self.data = {self.FileInfo['Data'][v][:] for v in self.vars}
    
    def calibrate(self):
        """
        辐射定标
        """
        self.data = {}
        for v in self.vars:
            nom = self.FileInfo['Data'][v]
            cal = self.FileInfo['Calibration'][v.replace('NOM', 'CAL')]
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
            self.calibrated_data[v] = target_channel
        return self.calibrated_data

class fy4b_giirs(fy4_L1):
    """
    读取风云4B-GIIRS的L1数据
    """
    def __init__(self, fpath):
        super().__init__(fpath)
        self.read_data()
        self.LW_wavelength = self.FileInfo['Data']['WN_LW'][:]
        self.MW_wavelength = self.FileInfo['Data']['WN_MW'][:]

    def __getitem__(self, varname):
        return self.data[varname]

    def read_data(self):
        """
        读取辐射数据
        """
        self.vars = ['VIS', 'NEdRLW', 'NEdRMW', 'RealLW', 'RealMW', 'ImaginaryLW', 'ImaginaryMW']
        
        # 读取可见光数据并定标
        self.data['VIS'] = {'data':self.FileInfo['Data']['VIS_DN'][:],
                            'dims':['line', 'column'],
                            'latitude':self.FileInfo['Geolocation']['Latitude_VIS'][:],
                            'longitude':self.FileInfo['Geolocation']['Longitude_VIS'][:],
                            'wavelength':'single',
                            }
        # 读取长波红外数据
        self.data['NEdRLW'] = {'data':self.FileInfo['Data']['NEdR_LW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['Geolocation']['Latitude_LW'][:],
                            'longitude':self.FileInfo['Geolocation']['Longitude_LW'][:],
                            'wavelength':self.FileInfo['Data']['WN_LW'][:],
                            }
        self.data['RealLW'] = {'data':self.FileInfo['Data']['ES_RealLW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['Geolocation']['Latitude_LW'][:],
                            'longitude':self.FileInfo['Geolocation']['Longitude_LW'][:],
                            'wavelength':self.FileInfo['Data']['WN_LW'][:],
                            }
        self.data['ImaginaryLW'] = {'data':self.FileInfo['Data']['ES_ImaginaryLW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['Geolocation']['Latitude_LW'][:],
                            'longitude':self.FileInfo['Geolocation']['Longitude_LW'][:],
                            'wavelength':self.FileInfo['Data']['WN_LW'][:],
                            }
        # 读取中波红外数据
        self.data['NEdRMW'] = {'data':self.FileInfo['Data']['NEdR_MW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['Geolocation']['Latitude_MW'][:],
                            'longitude':self.FileInfo['Geolocation']['Longitude_MW'][:],
                            'wavelength':self.FileInfo['Data']['WN_MW'][:],
                            }
        self.data['RealMW'] = {'data':self.FileInfo['Data']['ES_RealMW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['Geolocation']['Latitude_MW'][:],
                            'longitude':self.FileInfo['Geolocation']['Longitude_MW'][:],
                            'wavelength':self.FileInfo['Data']['WN_MW'][:],
                            }
        self.data['ImaginaryMW'] = {'data':self.FileInfo['Data']['ES_ImaginaryMW'][:],
                            'dims':['wavelength', 'point'],
                            'latitude':self.FileInfo['Geolocation']['Latitude_MW'][:],
                            'longitude':self.FileInfo['Geolocation']['Longitude_MW'][:],
                            'wavelength':self.FileInfo['Data']['WN_MW'][:],
                            }
    def calibrate(self):
        """
        对FY4B/GIIRS的可见光数据进行定标

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
        self.calbrated_data = {}
        # 读取数据集
        nom = self.FileInfo['Data']['ES_ContVIS']
        cal = self.FileInfo['Data']['ES_CalSTableVIS']
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