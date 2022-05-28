from util.data_util import convert_format

def run_en(eng,x,x1):
    x = convert_format(x)
    x1 = convert_format(x1)
    ret = eng.elno(x,x1)
    return ret["Elkan_Noto"]

