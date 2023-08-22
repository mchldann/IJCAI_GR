import matplotlib.pyplot as plt
import numpy as np

class Dialog(object):

    def __init__(self):
        self.reset()


    def reset(self):
        self.plots = {}
        self.plot_width = 200
        self.first_update = True
        self.num_plots = None


    def add_data_point(self, plot_name, x_value, y_values, trim_x, show_graphs):

        if plot_name not in self.plots:

            self.plots[plot_name] = {}

            self.plots[plot_name]["fig"] = plt.figure(num=plot_name)

            self.plots[plot_name]["fig"].canvas.draw()

            if show_graphs:
                plt.show(block=False)

            self.plots[plot_name]["x"] = np.empty([1, 1])

            self.num_plots = len(y_values)
            i = 0
            while i < self.num_plots:
                self.plots[plot_name]["y" + str(i)] = np.empty([1, 1])
                i += 1

            self.plots[plot_name]["min_y"] = y_values[0]
            self.plots[plot_name]["max_y"] = y_values[0] + 0.00001
            self.plots[plot_name]["trim_x"] = trim_x

        plot = self.plots[plot_name]
        plot["x"] = np.append(plot["x"], x_value)

        i = 0
        while i < self.num_plots:

            if y_values[i] < plot["min_y"]:
                plot["min_y"] = y_values[i]

            if y_values[i] > plot["max_y"]:
                plot["max_y"] = y_values[i]

            plot["y" + str(i)] = np.append(plot["y" + str(i)], y_values[i])

            i += 1

        if trim_x and len(plot["x"]) > self.plot_width:

            plot["x"] = plot["x"][1:(self.plot_width + 1)]

            i = 0
            while i < self.num_plots:
                plot["y" + str(i)] = plot["y" + str(i)][1:(self.plot_width + 1)]
                i += 1


    def update_image(self, debug_txt, labels):

        for key in self.plots:

            plt.figure(num=key)

            plot = self.plots[key]

            self.plots[key]["fig"].canvas.set_window_title(debug_txt)

            self.plots[key]["fig"].canvas.draw()

            # To avoid warnings about xmin = xmax
            if self.first_update:
                plt.xlim(max(0, plot["x"][0]), plot["x"][0] + 0.00001)
                self.first_update = False
            else:
                plt.xlim(max(0, plot["x"][0]), max(0.00001, plot["x"][len(plot["x"]) - 1]))

            plt.ylim(plot["min_y"], plot["max_y"])

            i = 0
            while i < self.num_plots:
                plt.plot(plot["x"], plot["y" + str(i)], c='C' + str(i), label=labels[i])
                i += 1

            plt.legend(loc="upper left")
            plt.draw()
            plot["fig"].canvas.flush_events()

            plt.clf()


    def save_image(self, log_dir):

        for key in self.plots:

            plt.figure(num=key)

            plot = self.plots[key]

            self.plots[key]["fig"].canvas.draw()

            # To avoid warnings about xmin = xmax
            if self.first_update:
                plt.xlim(plot["x"][0], plot["x"][0] + 0.00001)
                self.first_update = False
            else:
                plt.xlim(plot["x"][0], plot["x"][len(plot["x"]) - 1])

            plt.ylim(plot["min_y"], plot["max_y"])

            i = 0
            while i < self.num_plots:
                plt.plot(plot["x"], plot["y" + str(i)], c='C' + str(i))
                i += 1

            plt.draw()
            plot["fig"].canvas.flush_events()
            plot["fig"].savefig(log_dir + 'plot_' + str(key) + '.png', format='png')
            plt.clf()
