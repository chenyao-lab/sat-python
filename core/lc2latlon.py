import numpy as np
from numpy import sin,cos,arctan,pi,sqrt

def lc2latlon(x, y, sat, **kwargs):
    """
    静止卫星行列号转经纬度

    Parameters
    -----
    x : int
        行号. 一维数组或列表, 长度等于坐标点的个数.
    y : int
        列号. 一维数组或列表, 长度等于坐标点的个数且应与x的长度相等.
    sat : str
        卫星名称. 'FY4' or 'Himawari8'.
    sub_lon : float
        星下点经度
    h : float
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
    resolution : int, optional
        仅当sat为'FY4'时有效. 空间分辨率. 250, 500, 1000, 2000, 4000. 默认值为250.


    Returns
    -----
    lon : 经度.
    lat : 纬度.
    """
    
    # 基本参数
    default_parameters = {
        "h": 42164,
        "ea": 6378.137,
        "eb": 6356.7523
    }
    default_parameters.update(kwargs)
    if sat == 'FY4' and 'resolution' in kwargs.keys():
        # 根据分辨率确定参数
        if kwargs['resolution'] in [250, '250']:
            default_parameters['COFF'] = 21983.5
            default_parameters['CFAC'] = 163730199
            default_parameters['LOFF'] = 21983.5 
            default_parameters['LFAC'] = 163730199 
        elif kwargs['resolution'] in [500, '500']:
            default_parameters['COFF'] = 10991.5 
            default_parameters['CFAC'] = 81865099 
            default_parameters['LOFF'] = 10991.5 
            default_parameters['LFAC'] = 81865099 
        elif kwargs['resolution'] in [1000, '1000']:
            default_parameters['COFF'] = 5495.5  
            default_parameters['CFAC'] = 40932549 
            default_parameters['LOFF'] = 5495.5 
            default_parameters['LFAC'] = 40932549 
        elif kwargs['resolution'] in [2000, '2000']:
            default_parameters['COFF'] = 2747.5  
            default_parameters['CFAC'] = 20466274  
            default_parameters['LOFF'] = 2747.5 
            default_parameters['LFAC'] = 20466274 
        elif kwargs['resolution'] in [4000, '4000']:
            default_parameters['COFF'] = 1373.5  
            default_parameters['CFAC'] = 10233137
            default_parameters['LOFF'] = 1373.5  
            default_parameters['LFAC'] = 10233137  
        else:
            raise ValueError("风云4号卫星L1数据的resolution参数应为250, 500, 1000, 2000或4000")
    h, ea, eb = default_parameters['h'], default_parameters['ea'], default_parameters['eb']

    row = np.array(y)
    col = np.array(x)

    x = pi / 180.0 * (col - default_parameters['COFF']) / (2**-16 * default_parameters['CFAC'])
    y = pi / 180.0 * (row - default_parameters['LOFF']) / (2**-16 * default_parameters['LFAC'])

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

        lon = 180 / pi * arctan(S2 / S1) + default_parameters['sub_lon']
        lat = 180 / pi * arctan((ea * ea) / (eb * eb) * S3 / Sxy)

    lat = np.array(lat, dtype=np.float32)
    lon = np.array(lon, dtype=np.float32)

    lon = np.where(lon>180, lon-360, lon)

    return lon, lat