from util.data_util import convert_format

def run_roc(eng, x,x1):
    x = convert_format(x)
    x1 = convert_format(x1)
    ret = eng.roc(x,x1)
    print("ret",ret)
    return float(ret)
    