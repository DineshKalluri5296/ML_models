import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

joblib.dump(model, "kyphosis.pkl")
print("✅ Model saved as kyphosis.pkl")
