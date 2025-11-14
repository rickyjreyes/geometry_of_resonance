import pandas as pd

# Load the uploaded WCT harmonic spectrum
df = pd.read_csv("C:/Users/Ricky.Reyes.CST/Desktop/photon/Photon/results/wct_harmonics.csv")

# Count how many known Standard Model particles were successfully matched
knowns_df = df[df["Known Particle"] != "?"]
num_known = knowns_df.shape[0]
total_harmonics = df.shape[0]

# Unique known particles identified
unique_known_particles = knowns_df["Known Particle"].unique()
num_unique_particles = len(unique_known_particles)

{
    "Total Harmonics": total_harmonics,
    "Harmonics with Known Particles": num_known,
    "Unique Known Particles Identified": num_unique_particles,
    "Particle Names": sorted(unique_known_particles.tolist())
}

print();