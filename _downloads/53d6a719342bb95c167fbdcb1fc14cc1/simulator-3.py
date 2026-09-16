from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

import matplotlib.pyplot as plt
from stingray import sampledata
from stingray.simulator import simulator

# Obtain a sample light curve
lc = sampledata.sample_data().counts
# Instantiate simulator object
sim = simulator.Simulator(N=1024, mean=0.5, dt=0.125, rms=1.0)
# Obtain an artificial impulse response
ir = sim.relativistic_ir()
# Simulate
lc_new = sim.simulate(lc, ir)

plt.plot(lc_new.counts, 'g')
plt.title('Impulse Response based Simulation', fontsize='16')
plt.xlabel('Counts', fontsize='14')
plt.ylabel('Flux', fontsize='14')
plt.show()