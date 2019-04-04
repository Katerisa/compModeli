import espressomd
print(espressomd.features())
required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)

# STEP 1 system setup

# Importing other relevant python modules
import numpy as np
# System parameters
from espressomd.io.writer import vtf
fp = open('trajectory.vtf', mode='w+t')
from espressomd import checkpointing

checkpoint = checkpointing.Checkpoint(checkpoint_id="mycheckpoint")

n_part = 1000
density = 0.7

box_l=np.power(n_part/density, 1.0/3.0)*np.ones(3)

system = espressomd.System(box_l=box_l)
system.seed = 42

skin = 0.4
time_step = 0.1
eq_tstep = 0.001
temperature = 0.728
checkpoint.register("skin")
checkpoint.register("eq_tstep")
checkpoint.register("temperature")

system.time_step = time_step

system.thermostat.turn_off()
checkpoint.register("system")

system.thermostat.set_langevin(kT=1.0, gamma=0.5)
checkpoint.register("system.thermostat")



# STEP 2 Placing and accessing particles

# Add particles to the simulation box at random positions

for i in range(int(n_part/2)):
    system.part.add(type=0, pos=np.random.random(3) * system.box_l)
    system.part.add(type=1, pos=np.random.random(3) * system.box_l)


# Obtain all particle positions
cur_pos = system.part[:].pos
checkpoint.register("system.part")

# STEP 3 Setting up non-bonded interactions

lj_eps = 1.0
lj_sig = 1.0
lj_cut = 2.5*lj_sig
lj_cap = 0.5
lj_cut_mixed = 2**(1/6) * lj_sig
system.non_bonded_inter[0, 0].lennard_jones.set_params(epsilon=lj_eps, sigma=lj_sig,cutoff=lj_cut, shift='auto')
system.non_bonded_inter[1, 1].lennard_jones.set_params(epsilon=lj_eps, sigma=lj_sig,cutoff=lj_cut, shift='auto')
system.non_bonded_inter[0, 1].lennard_jones.set_params(epsilon=lj_eps, sigma=lj_sig,cutoff=lj_cut_mixed, shift='auto')

system.force_cap=lj_cap

checkpoint.register("system.non_bonded_inter")

# write structure block as header
vtf.writevsf(system, fp, types='all')
# write initial positions as coordinate block
vtf.writevcf(system, fp, types = 'all')

# STEP 4 Warmup

warm_steps  = 100
warm_n_time = 2000
min_dist    = 0.87

i = 0
act_min_dist = system.analysis.min_dist()
while i < warm_n_time and act_min_dist < min_dist :
    system.integrator.run(warm_steps)
    act_min_dist = system.analysis.min_dist()
    i+=1
    lj_cap += 1.0
    system.force_cap=lj_cap

# STEP 5 Integrating equations of motion and taking measurements

system.force_cap=0

# Integration parameters
sampling_interval = 50
sampling_iterations = 200

from espressomd.observables import ParticlePositions
from espressomd.accumulators import Correlator

# Pass the ids of the particles to be tracked to the observable.
part_pos = ParticlePositions(ids=range(n_part))
# Initialize MSD correlator
msd_corr = Correlator(obs1=part_pos,
                      tau_lin=10, delta_N=10,
                      tau_max=sampling_iterations * time_step,
                      corr_operation="square_distance_componentwise")
# Calculate results automatically during the integration
system.auto_update_accumulators.add(msd_corr)

# Set parameters for the radial distribution function
r_bins = 50
r_min = 0.0
r_max = system.box_l[0] / 2.0

avg_rdf00 = np.zeros((r_bins,))
avg_rdf11 = np.zeros((r_bins,))
avg_rdf01 = np.zeros((r_bins,))

# Take measurements
time = np.zeros(sampling_iterations)
instantaneous_temperature = np.zeros(sampling_iterations)
etotal = np.zeros(sampling_iterations)

max = 0
for i in range(1, sampling_iterations + 1):
    if i % (sampling_iterations / 100) == 0:
        print(i/float(sampling_iterations) * 100)
    system.integrator.run(sampling_interval)
    # Measure radial distribution function
    r00, rdf00 = system.analysis.rdf(rdf_type="rdf", type_list_a=[0], type_list_b=[0], r_min=r_min, r_max=r_max,r_bins=r_bins)
    r11, rdf11 = system.analysis.rdf(rdf_type="rdf", type_list_a=[1], type_list_b=[1], r_min=r_min, r_max=r_max,r_bins=r_bins)
    r01, rdf01 = system.analysis.rdf(rdf_type="rdf", type_list_a=[0], type_list_b=[1], r_min=r_min, r_max=r_max,r_bins=r_bins)
    avg_rdf00 += rdf00 / sampling_iterations
    avg_rdf11 += rdf11 / sampling_iterations
    avg_rdf01 += rdf01 / sampling_iterations

    # Measure energies
    energies = system.analysis.energy()
    kinetic_temperature = energies['kinetic'] / (1.5 * n_part)
    etotal[i - 1] = energies['total']
    time[i - 1] = system.time
    instantaneous_temperature[i - 1] = kinetic_temperature
    vtf.writevcf(system, fp)

    for pt in system.part.select(lambda p: True):
        force = (pt.f[0]**2 + pt.f[1]**2 + pt.f[2]**2)**(1/2)
        if force > max:
            max = force


print(max)
from espressomd import electrostatics
p3m = electrostatics.P3M(prefactor=1.0, accuracy=1e-2)
system.actors.add(p3m)
checkpoint.register("p3m")

fp.close()

# Finalize the correlator and obtain the results
msd_corr.finalize()
msd = msd_corr.result()

# STEP 6

import matplotlib.pyplot as plt
fig1 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
fig1.set_tight_layout(False)
plt.plot(r00, avg_rdf00,'-', color="#A60628", linewidth=2, alpha=1)
plt.plot(r11, avg_rdf11,'-', color="#1528b5", linewidth=2, alpha=1)
plt.plot(r01, avg_rdf01,'-', color="#0dbf22", linewidth=2, alpha=1)
plt.xlabel('$r$',fontsize=20)
plt.ylabel('$g(r)$',fontsize=20)
plt.show()

def second_graph():
    fig2 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    fig2.set_tight_layout(False)
    plt.plot(time, instantaneous_temperature,'-', color="red", linewidth=2, alpha=0.5, label='Instantaneous Temperature')
    plt.plot([min(time),max(time)], [temperature]*2,'-', color="#348ABD", linewidth=2, alpha=1, label='Set Temperature')
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Temperature',fontsize=20)
    plt.legend(fontsize=16,loc=0)
    plt.show()

def third_graph():
    fig3 = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    fig3.set_tight_layout(False)
    plt.plot(msd[:,0], msd[:,2]+msd[:,3]+msd[:,4],'-o', color="#348ABD", linewidth=2, alpha=1)
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Mean squared displacement',fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

# calculate the standard error of the mean of the total energy
standard_error_total_energy=np.sqrt(etotal.var())/np.sqrt(sampling_iterations)
print(standard_error_total_energy)

print(system.part[id].f)