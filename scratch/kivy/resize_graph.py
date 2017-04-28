'''
Created on 06.10.2016

@author: Yingxiong
'''
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout


class Main(App):

    def build(self):
        root = BoxLayout()
        self.graph = Graph(
            y_grid_label=False, x_grid_label=False, padding=5,
            xmin=0, xmax=100, ymin=0, ymax=30)

        line = MeshLinePlot(points=[(0, 0), (100, 30)])
        self.graph.add_plot(line)

        root.add_widget(self.graph)
        return root

if __name__ == '__main__':
    Main().run()
