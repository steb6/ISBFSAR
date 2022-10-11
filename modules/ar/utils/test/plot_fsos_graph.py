from matplotlib import pyplot as plt
import numpy as np
import pickle

# {'2500': {'FSOS-ACC': '0.67±0.06', 'FS-ACC': '0.84±0.08', 'OS-ACC': '0.70±0.07'}}
# {'3000': {'FSOS-ACC': '0.64±0.07', 'FS-ACC': '0.80±0.10', 'OS-ACC': '0.68±0.05'}}

with open("RESULTS_VALIDATION", "rb") as f:
# with open("RESULTS_3000_10_REP", "rb") as f:
    results = pickle.load(f)

for checkpoint in results.keys():
    for metric in results[checkpoint].keys():
        if metric == "OS-F1":
            continue
        std = round(np.std(results[checkpoint][metric][0]), 2)
        results[checkpoint][metric] = round(sum(results[checkpoint][metric][0]) / len(results[checkpoint][metric][0]), 2)
        results[checkpoint][metric] = str(results[checkpoint][metric]) + "±" + str(std)

print(results)
exit()
metric = "FSOS-ACC"  # OS-ACC FSOS-ACC OS-F1

# Data
disc_accs = np.array(results["DISC"][metric])
disc_cfs = np.std(disc_accs, axis=1)
disc_accs = disc_accs.mean(axis=1)

exp_accs = np.array(results["EXP"][metric])
exp_cfs = np.std(exp_accs, axis=1)
exp_accs = exp_accs.mean(axis=1)

discnoos_accs = np.array(results["DISC-NO-OS"][metric])
discnoos_cfs = np.std(discnoos_accs, axis=1)
discnoos_accs = discnoos_accs.mean(axis=1)

ks = np.linspace(5, 16, 12)

# Plot
fig, ax = plt.subplots()

ax.plot(ks,disc_accs, 'g--', label="TRX-OS")
ax.fill_between(ks, (disc_accs-disc_cfs), (disc_accs+disc_cfs), color='g', alpha=.1)

ax.plot(ks,exp_accs, 'r:', label="EXP")
ax.fill_between(ks, (exp_accs-exp_cfs), (exp_accs+exp_cfs), color='r', alpha=.1)

# ax.plot(ks,discnoos_accs, 'b-', label="DISC-NO-OS")
# ax.fill_between(ks, (discnoos_accs-discnoos_cfs), (discnoos_accs+discnoos_cfs), color='b', alpha=.1)

# Show
plt.xlabel("K - dimension of support set")
plt.ylabel("FSOS accuracy")
plt.legend()
plt.show()
