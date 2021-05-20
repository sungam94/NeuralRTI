import numpy as np


def load_lp(data_path, filename="dirs.lp"):
    f = open(data_path + '/' + filename)
    data = f.read()
    f.close
    linesn = data.split('\n')
    numLight = int(linesn[0])
    lines = linesn[1:]

    L = np.zeros((numLight, 3), np.float32)

    #### read light directions
    for i, l in enumerate(lines):
        s = l.split(" ")
        if len(s) == 4:
            L[i, 0] = float(s[1])
            L[i, 1] = float(s[2])
            L[i, 2] = float(s[3])
    return(L)


if __name__=='__main__':
    data_path = "/home/mk301/RTI/loewenkopf"
    print(load_lp(data_path))
