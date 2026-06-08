# ========================================================
# LNC-SCOPE FULL PIPELINE
# PART 1: Custom lncRNA Folding + Accessibility
# PART 2: miRNA Binding + Functional Prediction
# =========================================================

import json
import time
import csv
import os

# =========================================================
# PART 1 : CUSTOM RNA FOLDING
# =========================================================
# -------------------------------
# ENERGY MODEL
# -------------------------------

def pair_energy(a, b):
    if (a == 'G' and b == 'C') or (a == 'C' and b == 'G'):
        return -3
    elif (a == 'A' and b == 'U') or (a == 'U' and b == 'A'):
        return -2
    elif (a == 'G' and b == 'U') or (a == 'U' and b == 'G'):
        return -1
    return 0


def stacking_bonus(seq, i, j):
    bonus = 0

    if i + 1 < j - 1:
        if pair_energy(seq[i + 1], seq[j - 1]) < 0:
            bonus -= 1.5

    if i - 1 >= 0 and j + 1 < len(seq):
        if pair_energy(seq[i - 1], seq[j + 1]) < 0:
            bonus -= 1.0

    return bonus

# -------------------------------
# PARAMETERS
# -------------------------------

MIN_LOOP = 3
PAIR_BIAS = -0.3
UNPAIRED_PENALTY = 0.05


def loop_penalty(length):
    return 0.2 + 0.02 * length

# -------------------------------
# DP FOLDING
# -------------------------------

def fold_rna_energy(seq):
    n = len(seq)
    dp = [[0 for _ in range(n)] for _ in range(n)]

    for length in range(1, n):
        for i in range(n - length):
            j = i + length

            if j - i <= MIN_LOOP:
                dp[i][j] = 0
                continue

            best = dp[i + 1][j] + UNPAIRED_PENALTY
            best = min(best, dp[i][j - 1] + UNPAIRED_PENALTY)

            e = pair_energy(seq[i], seq[j])

            if e < 0:
                loop_len = j - i - 1
                penalty = loop_penalty(loop_len)
                stack = stacking_bonus(seq, i, j)

                best = min(
                    best,
                    dp[i + 1][j - 1] + e + penalty + PAIR_BIAS + stack
                )

            for k in range(i, j):
                best = min(best, dp[i][k] + dp[k + 1][j])

            dp[i][j] = best

    return dp


def traceback_energy(dp, seq, i, j, structure):
    if i >= j:
        return

    if dp[i][j] == dp[i + 1][j] + UNPAIRED_PENALTY:
        traceback_energy(dp, seq, i + 1, j, structure)
        return

    if dp[i][j] == dp[i][j - 1] + UNPAIRED_PENALTY:
        traceback_energy(dp, seq, i, j - 1, structure)
        return

    e = pair_energy(seq[i], seq[j])

    if e < 0:
        loop_len = j - i - 1
        penalty = loop_penalty(loop_len)
        stack = stacking_bonus(seq, i, j)

        if dp[i][j] == dp[i + 1][j - 1] + e + penalty + PAIR_BIAS + stack:
            structure[i] = '('
            structure[j] = ')'
            traceback_energy(dp, seq, i + 1, j - 1, structure)
            return

    for k in range(i, j):
        if dp[i][j] == dp[i][k] + dp[k + 1][j]:
            traceback_energy(dp, seq, i, k, structure)
            traceback_energy(dp, seq, k + 1, j, structure)
            return


def fold_rna(seq):
    dp = fold_rna_energy(seq)
    structure = ['.'] * len(seq)

    if len(seq) > 0:
        traceback_energy(dp, seq, 0, len(seq) - 1, structure)

    return ''.join(structure)


# -------------------------------
# ACCESSIBLE REGIONS (6-mer seed)
# -------------------------------

def find_accessible_regions(structure, seed_length=6, min_open=4):
    regions = []

    for i in range(len(structure) - seed_length + 1):
        window = structure[i:i+seed_length]

        if window.count('.') >= min_open:
            regions.append({
                "start": i,
                "end": i + seed_length,
                "length": seed_length,
                "open_count": window.count('.')
            })

    return regions


# -------------------------------
# FUNCTIONAL ZONES
# -------------------------------

def classify_functional_zones(regions):
    zones = []

    for region in regions:
        l = region["length"]

        if l >= 10:
            zone_type = "Sponge Zone"
        elif l <= 4:
            zone_type = "Scaffold Candidate"
        else:
            zone_type = "Hybrid Zone"

        zones.append({
            "start": region["start"],
            "end": region["end"],
            "type": zone_type
        })

    return zones

# -------------------------------
# MUTATION SENSITIVITY
# -------------------------------

def mutate_base(base):
    return {'A': 'U', 'U': 'G', 'G': 'C'}.get(base, 'A')


def mutation_sensitivity(seq):
    if not seq:
        return []

    original = fold_rna(seq)
    sensitive_positions = []

    for i in range(0, len(seq), 50):
        mutated = list(seq)
        mutated[i] = mutate_base(seq[i])

        if fold_rna(''.join(mutated)) != original:
            sensitive_positions.append(i)

    return sensitive_positions

# =========================================================
# PART 2 : miRNA MODULE
# =========================================================

def load_mirna_csv(file_path):
    mirna_dict = {}

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row['name']
            seq = row['sequence'].upper().replace("T", "U")

            if len(seq) >= 7:
                mirna_dict[name] = seq

    return mirna_dict


def is_pair(a, b):
    return (
        (a == 'A' and b == 'U') or
        (a == 'U' and b == 'A') or
        (a == 'G' and b == 'C') or
        (a == 'C' and b == 'G') or
        (a == 'G' and b == 'U') or
        (a == 'U' and b == 'G')
    )


def count_pairs(seq1, seq2):
    return sum(1 for a, b in zip(seq1, seq2) if is_pair(a, b))


def reverse_complement(seq):
    comp = {'A':'U','U':'A','G':'C','C':'G'}
    return ''.join(comp[b] for b in seq[::-1])


def build_kmer_set(sequence, k=6):
    return {sequence[i:i+k] for i in range(len(sequence) - k + 1)}


def prefilter_mirna(sequence, miRNA_data):
    kmer_set = build_kmer_set(sequence, 6)
    filtered = {}

    for name, seq in miRNA_data.items():
        seed = seq[1:7]
        seed_rc = reverse_complement(seed)

        if seed_rc in kmer_set:
            filtered[name] = seq

    return filtered

def find_matches(sequence, regions, miRNA_data):
    matches = []

    for name, mir_seq in miRNA_data.items():
        seed = mir_seq[1:7]

        for region in regions:
            sub_seq = sequence[region["start"]:region["end"]]

            for i in range(len(sub_seq) - len(seed) + 1):
                window = sub_seq[i:i+len(seed)]

                if count_pairs(window, seed) >= 5:
                    matches.append({
                        "miRNA": name,
                        "position": region["start"] + i,
                        "seed": seed,
                        "window": window
                    })

    return matches


def get_pair_energy(a, b):
    pair = a + b

    energy_table = {
        "AU": -1.1, "UA": -1.1,
        "GC": -2.3, "CG": -2.3,
        "GU": -0.9, "UG": -0.9
    }

    return energy_table.get(pair, 0)


def calc_duplex_energy(seq1, seq2):
    return sum(get_pair_energy(a, b) for a, b in zip(seq1, seq2))


def score_matches(matches, sequence):
    results = []

    for m in matches:
        start = m["position"]
        length = len(m["seed"])

        target_seq = sequence[start:start+length]

        deltaG = calc_duplex_energy(target_seq, m["seed"])

        if deltaG < -3:
            results.append({
                "miRNA": m["miRNA"],
                "pos": start,
                "deltaG": deltaG,
                "site": target_seq
            })

    return results


def classify_sponge(interactions):
    region_counts = {}

    for i in interactions:
        key = i["pos"] // 30
        region_counts[key] = region_counts.get(key, 0) + 1

    for v in region_counts.values():
        if v >= 2:
            return "miRNA sponge"

    return "non-sponge"


# =========================================================
# MAIN PROGRAM
# =========================================================

pipeline_start = time.time()

print("=== LNC-SCOPE PIPELINE STARTED ===")

# -------------------------------
# INPUT
# -------------------------------

choice = raw_input("Do you have dot-bracket structure? (yes/no): ").strip().lower()

sequence = ""
structure = ""

if choice == "yes":
    structure = raw_input("Paste dot-bracket structure:\n").strip()

else:
    file_choice = raw_input("Do you have FASTA file input? (yes/no): ").strip().lower()

    if file_choice == "yes":
        path = raw_input("Enter FASTA file path:\n").strip().strip('"')

        with open(path, "r") as f:
            lines = f.readlines()

        sequence = ''.join([l.strip() for l in lines if not l.startswith(">")])
        sequence = sequence.upper().replace("T", "U")

        print("\nSequence preview:")
        print(sequence[:60] + "...")

    else:
        sequence = raw_input("Paste RNA sequence:\n").strip().upper()
        sequence = sequence.replace("T", "U")

    print("\nSequence preview:")
    print(sequence[:60] + "...")

    # TIMING ADDED HERE (CORRECT PLACE)
    start = time.time()
    test = fold_rna(sequence)
    single_fold = time.time() - start

    print("\nSingle fold time:", round(single_fold, 2), "sec")
    print("Estimated mutation time:", round((single_fold * (len(sequence)//10))/60, 2), "minutes")

    structure = test

    print("\nPredicted structure:")
    print(structure)

# -------------------------------
# ANALYSIS
# -------------------------------

accessible_regions = find_accessible_regions(structure)
functional_zones = classify_functional_zones(accessible_regions)
sensitive_sites = mutation_sensitivity(sequence)

# Save structural stage
with open("structure_output.json", "w") as f:
    json.dump({
        "sequence": sequence,
        "structure": structure,
        "accessible_regions": accessible_regions,
        "functional_zones": functional_zones,
        "mutation_sensitive_positions": sensitive_sites
    }, f, indent=4)

print("\nStructure stage complete.")

# =========================================================
# miRNA STAGE
# =========================================================

miRNA_data = load_mirna_csv("mirna_sequences.csv")

print("Total miRNAs:", len(miRNA_data))

miRNA_data = prefilter_mirna(sequence, miRNA_data)

print("After filtering:", len(miRNA_data))

matches = find_matches(sequence, accessible_regions, miRNA_data)
scored = score_matches(matches, sequence)
function = classify_sponge(scored)

confidence = min(1.0, len(scored) / 5.0)
hotspots = [i["pos"] for i in scored]

final_output = {
    "lncRNA": "custom_input",
    "function": function,
    "miRNA": list(set(i["miRNA"] for i in scored)),
    "binding_sites": hotspots,
    "confidence": confidence,
    "accessible_regions": accessible_regions,
    "functional_zones": functional_zones
}

# -------------------------------
# FINAL SAVE
# -------------------------------

with open("final_lnc_scope_output.json", "w") as f:
    json.dump(final_output, f, indent=4)

# -------------------------------
# RESULTS
# -------------------------------

print("\n=== FINAL OUTPUT ===")
print(json.dumps(final_output, indent=4))

print("\nMatches found:", len(matches))
print("High-confidence interactions:", len(scored))

print("\nTotal pipeline time:",
      round(time.time() - pipeline_start, 2),
      "seconds")
