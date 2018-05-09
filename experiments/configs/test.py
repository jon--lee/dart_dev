import numpy as np
import yaml
import IPython

f = open ('linear.yaml')
data = yaml.load(f)
IPython.embed()

f.close()


