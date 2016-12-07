import os
import matplotlib
matplotlib.use("PDf")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

class TrainingProcess():
    def __init__(self, figure_data):
        print 'training_process'
        self.datas = figure_data
        self.number_of_figures = np.array(figure_data).shape[1]
        self.dim_of_figures = [len(np.array(i).shape)+1 for i in figure_data[0]]
        print self.dim_of_figures
        self.frame_interval_tunning = 1
        self.fig = plt.figure()
        var = locals()

        for i in range(self.number_of_figures):
            data = np.array(self.datas)[:, i]
            if self.dim_of_figures[i] == 1:
                xlimit = (-1, len(data)+1)
                ylimit = (data.min()*0.95, data.max()*1.05)
                var['ax'+str(i+1)] = self.fig.add_subplot((self.number_of_figures+1)/2, 2, i+1,xlim=xlimit, ylim=ylimit)
                self.__dict__['line' + str(i+1)], = eval('ax'+str(i+1)).plot(0, 0)
            elif self.dim_of_figures[i] == 2:
                xlimit = (-1, len(data[0])+1)
                ylimit = (np.array([d.min() for d in data]).min()*0.95,
                          np.array([d.max() for d in data]).max()*1.05)
                var['ax'+str(i+1)] = self.fig.add_subplot((self.number_of_figures+1)/2, 2, i+1,xlim=xlimit, ylim=ylimit)
                self.__dict__['line' + str(i+1)], = eval('ax'+str(i+1)).plot([], [])
            else:
                xlimit = (-1, len(data[0])+1)
                ylimit = (-1, len(np.array(data[0]).T)+1)
                zlimit = (np.array([d.min() for d in data]).min()*0.95,
                          np.array([d.max() for d in data]).max()*1.05)

                var['ax'+str(i+1)] = self.fig.add_subplot((self.number_of_figures+1)/2, 2, i+1,xlim=xlimit, ylim=ylimit,zlim=zlimit, projection='3d')
                self.__dict__['line' + str(i+1)], = eval('ax'+str(i+1)).plot([0], [0], [0])

    def begin(self):
        lines = []
        for i in range(self.number_of_figures):
            data = np.array(self.datas)[:, i]
            d = len(np.array(data[0]).shape) + 1
            fun = 'self.start'+str(d)+'D'
            lines.append(eval(fun)(data, i+1))
        return lines


    def update(self, i):
        lines = []
        for j in range(self.number_of_figures):
            data = np.array(self.datas)[:, j]
            d = len(np.array(data[0]).shape) + 1
            fun = 'self.animate'+str(d)+'D'
            lines.append(eval(fun)(data, i/self.frame_interval_tunning, j+1))
        return lines


    def start2D(self,data,figure_index):
        print '2'
        line = eval('self.line'+str(figure_index))
        line.set_data([0],[0])
        return line

    def start3D(self,data,figure_index):
        print '3'
        line = eval('self.line'+str(figure_index))
        line.set_data([0],[0])
        line.set_3d_properties([0])
        return line

    def start1D(self,data,figure_index):
        print '1'
        line = eval('self.line'+str(figure_index))
        line.set_data(0, 0)
        return line

    def animate2D(self,data,i,figure_index):
        print '2nice'
        line = eval('self.line'+str(figure_index))
        line.set_data(range(len(data[i])), data[i])
        return line

    def animate3D(self,data,i,figure_index):
        print '3nice'
        x=[]
        y=[]
        z=[]
        for k in range(len(data[i])):
            for j in range(len(data[i][k])):
                x.append(k)
                y.append(j)
                z.append(data[i][k][j])
        line = eval('self.line'+str(figure_index))
        line.set_data(x,y)
        line.set_3d_properties(z)
        return line
    def animate1D(self,data,i,figure_index):
        print '1nice'
        x = range(10)
        y = [data[i]]*10
        line = eval('self.line'+str(figure_index))
        line.set_data(x, data[i])
        return line


    def run(self):
        length = self.frame_interval_tunning*len(self.datas)
        anim1 = animation.FuncAnimation(self.fig, self.update, init_func=self.begin,
                                        frames=length, interval=10)
        plt.show()


if __name__ == '__main__':
    figure_data = [[[0.03,0.03],[0.03,0.03],[0.03,0.03],0.11,[[0.01,0.01],[0.03,0.01]], [[0.01,0.01],[0.03,0.01],[0.01,0.01],[0.02,0.01]],
                    [0.01,0.03], [0.01,0.03,0.01,0.02]],
                   [[0.03,0.03],[0.03,0.03],[0.03,0.03],0.10,
                    [[0.01,0.01],[0.03,0.01]], [[0.01,0.02],[0.03,0.01],[0.01,0.02],[0.05,0.01]],[0.04,0.03], [0.01,0.07,0.01,0.02]]]
    d = TrainingProcess(figure_data)
    #d.begin()
    d.run()