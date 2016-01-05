import numpy as np

image = np.ma.array([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]],
                    mask = [[False, False, False],
                            [False, False, False],
                            [False, True, True]]
                    )

psf1 = np.ma.array([[1., 2., 3.],
                    [4., 5., 100.],
                    [7., 8., 9.]],
                    mask = [[False, False, False],
                            [False, False, True],
                            [False, False, False]]
                    )

psf2 = np.ma.array([[1., 2., 100.],
                    [4., 5., 6.],
                    [7., 8., 100.]],
                   mask = [[False, False, True],
                           [False, False, False],
                           [False, False, True]]
                    )
psfarray = np.ma.dstack((psf1, psf2))
