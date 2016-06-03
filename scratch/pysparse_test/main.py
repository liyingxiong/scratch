'''
Created on 26.04.2016

@author: Yingxiong
'''
from kivy.app import App
from pysparse.direct import superlu
import numpy as np
from pysparse import spmatrix
from kivy.uix.label import Label


class CanvasApp(App):

    @property
    def solved(self):
        A = spmatrix.ll_mat(5, 5)
        for i in range(5):
            A[i, i] = i + 1
        A = A.to_csr()
        B = np.ones(5)
        x = np.empty(5)
        LU = superlu.factorize(A, diag_pivot_thresh=0.0)
        LU.solve(B, x)
        return np.array_str(x)

        def build(self):
            return Label(text=self.solved)


if __name__ == '__main__':
    CanvasApp().run()
