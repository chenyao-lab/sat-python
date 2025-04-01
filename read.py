from core import FY4A, FY4B, Himawari8

if __name__ == '__main__':
    # example
    dataset = FY4A.giirs(r'D:\WorkSpace\MyProject\sat-python\examples\data\FY4A-_GIIRS-_N_REGX_1047E_L1-_IRD-_MULT_NUL_20240102231500_20240102232544_016KM_049V3.HDF')
    # dataset = FY4A.agri(r'D:\WorkSpace\ZokoProject\Changsha_Ocean\Data\FY4A\AGRI\FY4A-_AGRI--_N_REGX_0865E_L1-_FDI-_MULT_NOM_20241025001000_20241025001924_4000M_V0001.HDF')
    # dataset = Himawari8.himawari8_hsd(r'D:\WorkSpace\ZokoProject\Changsha_Ocean\Data\葵花8\HS_H08_20170623_0250_B01_FLDK\HS_H08_20170623_0250_B01_FLDK_R10_S0110.DAT.bz2')
    print(dataset.calibrate())