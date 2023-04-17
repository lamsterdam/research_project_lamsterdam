import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mydata = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length',
                                                      'petal_width', 'class'])