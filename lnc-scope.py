# =========================================================
# LNC-SCOPE FULL PIPELINE
# PART 1: Custom lncRNA Folding + Accessibility
# PART 2: miRNA Binding + Functional Prediction
# =========================================================

import json
import time
import csv
import os

from turner_folding import fold_turner, can_pair, mutation_sensitivity_fast
# =========================================================
# PART 1 : CUSTOM RNA FOLDING
# =========================================================

# -------------------------------
# TURNER FOLDING WRAPPER
# -------------------------------

def fold_rna(seq):
    structure, mfe = fold_turner(seq)
    fold_rna.last_mfe = mfe
    return structure


def fold_rna_with_energy(seq):
    return fold_turner(seq)

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

def merge_accessible_regions(regions):
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: r["start"])
    merged = []
    current = {"start": regions[0]["start"], "end": regions[0]["end"]}

    for r in regions[1:]:
        if r["start"] <= current["end"]:
            current["end"] = max(current["end"], r["end"])
        else:
            merged.append(current)
            current = {"start": r["start"], "end": r["end"]}
    merged.append(current)

    for m in merged:
        m["length"] = m["end"] - m["start"]

    return merged


# -------------------------------
# FUNCTIONAL ZONES
# -------------------------------

def classify_functional_zones(regions, seed_length=7):
    zones = []

    for region in regions:
        l = region["length"]

        if l >= seed_length * 3:
            zone_type = "Sponge Zone"
        elif l < seed_length:
            zone_type = "Scaffold Candidate"
        else:
            zone_type = "Hybrid Zone"

        zones.append({
            "start": region["start"],
            "end": region["end"],
            "length": l,
            "type": zone_type
        })

    return zones


# -------------------------------
# MUTATION SENSITIVITY
# -------------------------------

def mutate_base(base):
    return {'A': 'U', 'U': 'G', 'G': 'C'}.get(base, 'A')

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

choice = input("Do you have dot-bracket structure? (yes/no): ").strip().lower()

sequence = ""
structure = ""

if choice == "yes":
      structure = input("Paste dot-bracket structure: ")
      mfe = float(input("Enter MFE from RNAfold (kcal/mol): "))
      sequence = input("Paste the corresponding RNA sequence:\n").strip().upper()
      sequence = sequence.replace("T", "U")
      if len(sequence) != len(structure):
          print(f"WARNING: sequence length ({len(sequence)}) != structure length ({len(structure)}) — indices will misalign.")
else:
    file_choice = input("Do you have FASTA file input? (yes/no): ").strip().lower()

    if file_choice == "yes":
        path = input("Enter FASTA file path:\n").strip().strip('"')

        with open(path, "r") as f:
            lines = f.readlines()

        sequence = ''.join([l.strip() for l in lines if not l.startswith(">")])
        sequence = sequence.upper().replace("T", "U")

        print("\nSequence preview:")
        print(sequence[:60] + "...")

    else:
        sequence = input("Paste RNA sequence:\n").strip().upper()
        sequence = sequence.replace("T", "U")

# -------------------------------
# FOLD (only if structure wasn't already provided)
# -------------------------------

if choice != "yes":
    start = time.time()
    structure, mfe = fold_rna_with_energy(sequence)
    single_fold = time.time() - start

    print("\nSingle fold time:", round(single_fold, 2), "sec")

    print("\nPredicted structure (Turner 2004 model)")
    print("Predicted MFE:", round(mfe, 2), "kcal/mol")

# -------------------------------
# ANALYSIS
# -------------------------------

accessible_regions = find_accessible_regions(structure, seed_length=7, min_open=5)
merged_regions = merge_accessible_regions(accessible_regions)
functional_zones = classify_functional_zones(merged_regions, seed_length=7)
sensitive_sites = mutation_sensitivity_fast(sequence, window=80, min_delta=0.5)

# Save structural stage
with open("structure_output.json", "w") as f:
    json.dump({
        "sequence": sequence,
        "structure": structure,
        "mfe_kcal_mol": mfe,
        "accessible_regions": accessible_regions,
        "merged_regions": merged_regions,
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
    "mfe_kcal_mol": mfe,
    "function": function,
    "miRNA": list(set(i["miRNA"] for i in scored)),
    "binding_sites": hotspots,
    "confidence": confidence,
    "accessible_regions": accessible_regions,
    "merged_regions": merged_regions,
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

print("\n=== PIPELINE COMPLETE ===")
print("Function prediction:", final_output["function"])
print("Confidence:", round(final_output["confidence"], 2))
print("Results saved to final_lnc_scope_output.json")

print("\nMatches found:", len(matches))
print("High-confidence interactions:", len(scored))

print("\nTotal pipeline time:",
      round(time.time() - pipeline_start, 2),
      "seconds")
