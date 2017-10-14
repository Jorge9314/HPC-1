import numpy as np
import matplotlib.pyplot as plt
import sys

def readData(fileName):
    f = open(fileName,"r")
    data = f.readlines()

    dic = {} # dic[image] = [size, time]
    for i in range(1,len(data)):
        words = data[i].split()
        # print (words)
        key = words[1]
        if (key in dic):
            dic[key] = [float(words[0]), dic[key][1] + float(words[2])]
        else:
            dic[key] = [float(words[0]), float(words[2])]

    return dic

def getAxis(x, y, dic):
    for e in dic: # get the average of the times
        dic[e][1] = round(dic[e][1]/20, 3)
        # print (e + " -> ")
        # print (dic[e])
        x_aux.append(dic[e][1])
        y_aux.append(dic[e][0])

if __name__ == "__main__":

    # print (sys.argv)
    # print (len(sys.argv))
    if (len(sys.argv) != 3):
        print ("Usage: <times fileName> <plot title>")
        sys.exit()

    dic = readData(sys.argv[1])
    x_aux = []
    y_aux = []
    getAxis(x_aux, y_aux, dic)

    x = np.array(x_aux)
    y = np.array(y_aux)
    plt.xlabel('time (seconds)')
    plt.ylabel('size (bytes)')
    plt.scatter(x, y)
    # plt.plot(x, y)
    plt.yscale('linear')
    plt.title('Sobel Filter, ' + sys.argv[2])
    plt.grid(True)
    plt.show()
