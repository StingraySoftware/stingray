from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']

import matplotlib.pyplot as plt
from stingray.simulator import simulator

# Instantiate simulator object
sim = simulator.Simulator(N=1024, mean=0.5, dt=0.125, rms=1.0)
# Define a spectrum
w = np.fft.rfftfreq(sim.N, d=sim.dt)[1:]
spectrum = np.power((1/w),2/2)
# Simulate
lc = sim.simulate(spectrum)

plt.plot(lc.counts, 'g')
plt.title('User-defined Model Simulation', fontsize='16')
plt.xlabel('Counts', fontsize='14')
plt.ylabel('Flux', fontsize='14')
plt.show()