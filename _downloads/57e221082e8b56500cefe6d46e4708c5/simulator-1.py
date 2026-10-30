from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

import matplotlib.pyplot as plt
from stingray.simulator import simulator

# Instantiate simulator object
sim = simulator.Simulator(N=1024, mean=0.5, dt=0.125, rms=1.0)
# Specify beta value
lc = sim.simulate(2)

plt.plot(lc.counts, 'g')
plt.title('Random-walk Distribution Simulation', fontsize='16')
plt.xlabel('Counts', fontsize='14', )
plt.ylabel('Flux', fontsize='14')
plt.show()