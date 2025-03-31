# 此脚本用于读取Himawari-8卫星的HSD数据
import os
import bz2
import json
import numpy as np
from numpy import sin,cos,arctan,pi,sqrt

def lc2latlon_himawari8(x, y, **kwargs):
    """
    Himawari-8卫星行列号转经纬度

    Parameters
    -----
    x : int
        行号. 一维数组或列表, 长度等于坐标点的个数.
    y : int
        列号. 一维数组或列表, 长度等于坐标点的个数且应与x的长度相等.
    sub_lon : float, optional
        星下点经度
    h : float, optional
        卫星高度(km)
    CFAC : int
        列比例因子
    LFAC : int
        行比例因子
    COFF : int
        列偏移
    LOFF : int
        行偏移
    ea : float
        地球半长轴
    eb : float
        地球半短轴

    Returns
    -----
    lon : 经度.
    lat : 纬度.
    """
    
    # 基本参数
    default_parameters = {
        # "sub_lon": 140.7,
        # "CFAC": 40932549,
        # "LFAC": 40932549,
        # "COFF": 5500.5,
        # "LOFF": 5500.5,
        "h": 42164,
        "ea": 6378.137,
        "eb": 6356.7523
    }
    # 更新基本参数
    default_parameters.update(kwargs)
    # 获取参数
    lambda_d = default_parameters['sub_lon']
    CFAC = default_parameters['CFAC']
    LFAC = default_parameters['LFAC']
    COFF = default_parameters['COFF']
    LOFF = default_parameters['LOFF']
    h = default_parameters['h']
    ea = default_parameters['ea']
    eb = default_parameters['eb']
    
    
    row = np.array(y)
    col = np.array(x)

    x = pi / 180.0 * (col - COFF) / (2**-16 * CFAC)
    y = pi / 180.0 * (row - LOFF) / (2**-16 * LFAC)

    with np.errstate(invalid='ignore'):
        #  临时忽略警告
        sd = sqrt(
            (h * cos(x) * cos(y)) ** 2
            - (cos(y) * cos(y) + (ea * ea) / (eb * eb) * sin(y) * sin(y)) 
            * ((h * h) - (ea * ea))
            )

        sn = (h * cos(x) * cos(y) - sd) / (cos(y) * cos(y) + (ea * ea) / (eb * eb) * sin(y) * sin(y))

        S1 = h - (sn * cos(x) * cos(y))
        S2 = sn * sin(x) * cos(y)
        S3 = -sn * sin(y)
        Sxy = sqrt(S1 * S1 + S2 * S2)

        lon = 180 / pi * arctan(S2 / S1) + lambda_d
        lat = 180 / pi * arctan((ea * ea) / (eb * eb) * S3 / Sxy)

    lat = np.array(lat, dtype=np.float32)
    lon = np.array(lon, dtype=np.float32)

    lon = np.where(lon>180, lon-360, lon)

    return lon, lat

class himawari8_hsd:
    """
    读取并处理葵花8卫星数据
    """
    satellite = 'Himawari 8'
    product = 'HSD'
    vars = []
    Data = {}
    def __init__(self, fpath):
        fname = os.path.basename(fpath)
        self._fmt = fname.split('.')[-1]    # 文件格式
        self.satellite_name = fname.split('_')[1]   # 卫星名称
        self.observation_start_datetime = fname.split('_')[2] + fname.split('_')[3]   # 观测开始时间
        self.observation_area = fname.split('_')[5] # 观测区域
        self.band = int(fname.split('_')[4][1:]) # 波段号

        FileInfo = self.read_binfile(fpath)
        self.FileInfo = FileInfo
        sub_lon = FileInfo['Projection information']['sub_lon']
        CFAC = FileInfo['Projection information']['Column scaling factor']
        LFAC = FileInfo['Projection information']['Line scaling factor']
        COFF = FileInfo['Projection information']['Column offset']
        LOFF = FileInfo['Projection information']['Line offset']
        H = FileInfo['Projection information']["Distance from Earth's center to virtual satellite"]
        EA = FileInfo['Projection information']["Earth's equatorial radius"]
        EB = FileInfo['Projection information']["Earth's polar radius"]
        self.proj_args = {
            "sub_lon": sub_lon,
            "CFAC": CFAC,
            "LFAC": LFAC,
            "COFF": COFF,
            "LOFF": LOFF,
            "h": H,
            "ea": EA,
            "eb": EB
        }    
        self.lc2latlon()  
        self.calibrate_data()                              
                                               
    
    def __getitem__(self, varname):
        if varname in self.vars:
            return getattr(self, varname)
        else:
            raise KeyError(f"{varname} 不是有效数据变量!")
            
    def read_binfile(self, fpath):
        """
        从Himawari二进制文件中读取数据信息
        """
        if self._fmt == 'bz2':
            with bz2.open(fpath, 'rb') as f:
                info = f.read()
        elif self._fmt == 'DAT':
            with open(fpath, 'rb') as f:
                info = f.read()
        
        # 定义Himawari-8二进制格式化规则
        with open(os.path.join(os.path.dirname(__file__), '../config/Himawari_Standard_Data.json')) as f:
            hsd_form = json.load(f)
            formation = hsd_form['Formation']
            frame = hsd_form['Frame']
        # 根据观测区域筛选信息
        if self.observation_area == "FLDK":
            frame_area = frame['FullDisk']
        elif self.observation_area == "JPee":
            frame_area = frame['JapanArea']
        elif self.observation_area == "R3ff":
            frame_area = frame['TargetArea']
        else:
            frame_area = frame['LandmarkArea']
        # 根据波段筛选信息
        if self.band == 3:
            frame_area_band = frame_area['band3']
        elif self.band in [1,2,4]:
            frame_area_band = frame_area['band124']
        else:
            frame_area_band = frame_area['band5_16']
        EW_size, NS_size = frame_area_band['East-west direction'], frame_area_band['North-south direction']

        if self.band < 7:
            formation_band = formation['band1_6']
        else:
            formation_band = formation['band7_16']

        # 读取二进制数据并解码
        fileinfo = {}   # 用字典保存解码结果
        pos = 0 # 当前二进制流中读取到的位置索引
        block_length = np.nan   # 当前块的总字节数
        block_count = 0 # 当前块已经读取的字节数
        block_names = ['Basic information', 'Data information', 'Projection information', 'Navigation information', 'Calibration information', 'Inter-calibration information', 'Segment information', 'Navigation correction information', 'Observation time information', 'Error information', 'Spare', 'Data']
        block_index = -1
        for fm in formation_band:
            # 提取值的名称、类型代号、字节数
            name, dtype, byte_nums = fm
            # 将类型代号转换为dtype，并计算值所占的字节数
            if dtype == 'i1':
                length = 1 * byte_nums
                dtp = np.uint8
            elif dtype == 'i2':
                length = 2 * byte_nums
                dtp = np.uint16
            elif dtype == 'i4':
                length = 4 * byte_nums
                dtp = np.uint32
            elif dtype == 'C':
                length = 1 * byte_nums
            elif dtype == 'R4':
                length = 4 * byte_nums
                dtp = np.float32
            elif dtype == 'R8':
                length = 8 * byte_nums
                dtp = np.float64
            elif dtype == '':
                length = byte_nums*40
            else:
                raise

            if name.startswith('Block number'):
                block = {}  # 新字典保存当前块的信息
                block_index += 1
            
            if name == 'Count value of each pixel':
                # 此时定位到整个数据尾端，也是数值区域
                byte_nums = EW_size * NS_size
                length = byte_nums * 2
                value = np.frombuffer(info[pos:pos+length], dtype=dtp)
                fileinfo['Data'] = {name:value}

            if dtype == '':
                # 此时定位到block的末尾空白填充
                pos += (block_length - block_count) # 跳过空白填充
                block_count = 0 # block计数器重置为0
                fileinfo[block_names[block_index]] = block  # 将block添加到fileinfo
                continue
            elif dtype == 'C':
                value = np.char.decode(info[pos:pos+length], encoding='ascii')
                block[name] = value
            else:
                value = np.frombuffer(info[pos:pos+length], dtype=dtp)
                block[name] = value
                if name.startswith('Block length'):
                    block_length = value[0]

            pos += length
            block_count += length
            
        return fileinfo

    def lc2latlon(self):
        """
        将行列号转为经纬度坐标
        """
        columns = self.FileInfo["Data information"]["Number of columns"]
        lines = self.FileInfo["Data information"]["Number of lines"]
        first_line = self.FileInfo['Segment information']['First line number of image segment']
        xx, yy = np.meshgrid(np.arange(1, columns+1, 1), np.arange(first_line, first_line+lines, 1))
        lon, lat = lc2latlon_himawari8(xx.flatten(), yy.flatten(), **self.proj_args)
        self.lon = lon.reshape(xx.shape)
        self.lat = lat.reshape(yy.shape)
    
    def calibrate_data(self):
        """
        辐射定标, 1~6波段返回反射率, 7~16波段返回亮温
        """
        self.vars = ['radiance']    # 辐射率
        data = self.FileInfo['Data']['Count value of each pixel']    # 读取原始DN值
        gain = self.FileInfo['Calibration information']["Calibrated Slope for count-radiance conversion equation_updated value of No. 8 of this block"]
        offset = self.FileInfo['Calibration information']["Calibrated Intercept for count-radiance conversion equation_updated value of No. 9 of this block"]
        
        # 1~6波段定标为反射率，7~16波段定标为亮温
        if self.band < 7:
            # 根据辐射率计算反射率
            gain = self.FileInfo['Calibration information']["Calibrated Slope for count-radiance conversion equation_updated value of No. 8 of this block"]
            offset = self.FileInfo['Calibration information']["Calibrated Intercept for count-radiance conversion equation_updated value of No. 9 of this block"]
            # 计算辐射率
            self.radiance = data * gain + offset     # W / (m2 * sr * μm)
            self.albedo = self.radiance * self.FileInfo['Calibration information']["Coefficient for transformation from radiance  to albedo"]
        else:
            gain = self.FileInfo['Calibration information']["Slope for count-radiance conversion equation"]
            offset = self.FileInfo['Calibration information']["Intercept for count-radiance conversion equation"]
            self.radiance = data * gain + offset     # W / (m2 * sr * μm)
            # 根据辐射率计算亮温
            c0 = self.FileInfo['Calibration information']["radiance to brightness temperature_c0"]
            c1 = self.FileInfo['Calibration information']["radiance to brightness temperature_c1"]
            c2 = self.FileInfo['Calibration information']["radiance to brightness temperature_c2"]
            wl = self.FileInfo["central wave length"]
            h = self.FileInfo['Calibration information']["Planck constant"]
            k = self.FileInfo['Calibration information']["Boltzmann constant"]
            c = self.FileInfo['Calibration information']["Speed of light"]
            Te = h * c / k / wl / np.log(2*h*c*c/((wl**5)*self.radiance)+1) # 有效亮温
            self.Tb = c0 + c1 * Te + c2 * Te * Te    # 亮温
