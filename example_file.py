import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 100)  # 100 punktów od 0 do 10
y = np.sin(x)                # funkcja sinus

plt.plot(x, y)
plt.show()