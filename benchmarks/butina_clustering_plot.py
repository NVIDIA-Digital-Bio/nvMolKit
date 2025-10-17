import math
import matplotlib.pyplot as plt
plt.figure()
for cutoff in df['cutoff'].unique():
    subset = df[df['cutoff'] == cutoff]
    plt.errorbar(subset['size'], subset['rdkit_time_ms'], yerr=subset['rdkit_std_ms'] * 1.95 / math.sqrt(3), capsize=10, label=f'RDKit Cutoff {cutoff}', fmt='-o')
    plt.errorbar(subset['size'], subset['nvmol_time_ms'], yerr=subset['nvmol_std_ms']* 1.95 / math.sqrt(3), capsize=10, label=f'nvMolKit Cutoff {cutoff}', fmt='-o')
plt.xlabel('Number of Molecules')
plt.ylabel('Time (ms)')
plt.title('Butina Clustering Performance Comparison')
plt.legend()
plt.show()

# Speedup plot
plt.figure()
for cutoff in [0.1, 0.2]:
    #for cutoff in df['cutoff'].unique():
    subset = df[df['cutoff'] == cutoff]
    speedup = subset['rdkit_time_ms'] / subset['nvmol_time_ms']
    plt.plot(subset['size'], speedup, '-o', label=f'Cutoff {cutoff}')
plt.xlabel('Number of Molecules')
plt.ylabel('Speedup (RDKit / nvMolKit)')
plt.title('Butina Clustering Speedup')
plt.yscale('log')
plt.legend()
plt.show()

# Ratio of input size to time
plt.figure()
for cutoff in [0.1]:
    subset = df[df['cutoff'] == cutoff]
    rdkit_ratio = subset['size'] ** 2 / subset['rdkit_time_ms']
    nvmol_ratio = subset['size'] ** 2 / subset['nvmol_time_ms']
    plt.plot(subset['size'], rdkit_ratio, '-o', label=f'RDKit Cutoff {cutoff}')
    plt.plot(subset['size'], nvmol_ratio, '-o', label=f'nvMolKit Cutoff {cutoff}')
plt.legend()
plt.show()
