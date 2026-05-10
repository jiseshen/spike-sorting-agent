"""
Agent context and prompts for LLM-driven cluster curation.

Based on manual curation patterns from lab action sheets (CH3, CH20, CH30, CH31).
Key insight: Final outcome typically 1-2 large clusters after aggressive splitting/discarding.
"""

# =====================================================================
# NEURONAL CRITERIA - Core biological principles
# =====================================================================

NEURONAL_CRITERIA = """
## Valid Extracellular Action Potential Shape

A neuronal waveform must exhibit a standard extracellular spike morphology.

### Shape Requirements (Neuronal Check)

- **Phases:** Must be strictly biphasic or triphasic.
  - Expected pattern: small initial positive deflection → sharp negative trough → single smooth return toward baseline.
  - Invalid if the waveform shows >3 phases, extra bumps, secondary rises/falls, or multiple depolarization events.

- **Depolarization (Negative Peak):**
  - Should descend rapidly, forming a sharp and well-defined trough.
  - Should not be overly broad in time (peak-to-trough typically < 0.5–0.6 ms).
  - Should not contain pre-trough bumps or irregularities.

- **Repolarization:**
  - Should rise smoothly back toward baseline.
  - Should not include slow drifting, oscillations, or a second hump after the main trough.

- **Baseline Stability:**
  - Beginning and end of the waveform should be centered around baseline (near zero).
  - No large amplitude offsets or drift before or after the main spike.

However, natural waveform variability, vertical spread, or broadness from large cluster size 
does NOT count as baseline drift, DC offset, instability, or violation.
Only true morphology violations (monophasic noise, >3 phases, slow drift, or non-spike shape, etc.) 
should be treated as non-neuronal.
Do NOT hallucinate baseline shift or drift when the baseline is centered around zero.
Discard huge clusters (>10000) with caution.
"""

# =====================================================================
# DECISION CRITERIA - Extracted from manual curation
# =====================================================================


# =====================================================================
# QUALITY THRESHOLDS
# =====================================================================

QUALITY_THRESHOLDS = {
    # Spike count
    "auto_discard_threshold": 500,  # < 500 → automatic discard (no analysis)
    "min_spikes_standalone": 500,  # Minimum for standalone unit (post-split)
    "min_spikes_merge": 500,  # Below this, must merge or discard
    
    # ISI violations
    "max_isi_violation_rate": 0.006,  # 0.6% of ISI counts
    
    # Waveform consistency
    "max_amplitude_cv": 0.3,  # Coefficient of variation
    "max_amplitude_cv_large": 0.5,  # For large clusters (may contain embedded units)
    
    # Similarity
    "min_similarity_merge": 0.8,  # Waveform correlation for merging
    "min_similarity_hierarchy": 0.3,  # Low-similarity merges may be bad
    
    # Waveform shape
    "max_peak_to_trough_ms": 0.6,  # Maximum width (18 samples @ 30kHz)
    "min_peak_to_trough_ms": 0.15,  # Minimum width (too narrow = artifact)
    "max_phases": 3,  # Maximum number of phases (biphasic/triphasic only)
    "min_phases": 2,  # Minimum phases (monophasic = invalid)
    
    # Temporal drift
    "max_width_change_pct": 0.2,  # 20% change in width across time → drift
    
    # Cluster size classification
    "large_cluster_threshold": 500,  # Enter agent analysis
    "split_priority_threshold": 5000,  # Target: final valid clusters should have ≥5000 spikes
    "final_minimum_threshold": 5000,  # After all merges, discard clusters < 5000 spikes
}

# =====================================================================
# REASONING TEMPLATES
# =====================================================================

REASONING_TEMPLATES = {
    "split": """Cluster {cluster_id} waveforms have too much variability throughout (amplitude CV={amplitude_cv:.2f}); 
within its aggregation tree, we see that there are {n_subgroups} densely packed components that are each composed 
of numerous subclusters (total {n_overclusters} overclusters). This suggests {n_subgroups} embedded valid units. 
ISI violation rate is {isi_violation_rate:.1%}, which may improve after splitting.""",
    
    "discard_few_spikes": """Cluster {cluster_id} event count is too low ({n_spikes} spikes) for individual clusters 
at end of hierarchical merging. Insufficient data to establish reliable statistics or assess ISI quality.""",
    
    "discard_shape_violation": """Cluster {cluster_id} has {shape_issue} (e.g., too many phases, too broad, slow descent). 
The waveform does not fit physiological expectations for an extracellular recording from a single neuron. 
Additionally, it has {n_spikes} spikes with {isi_violation_rate:.1%} ISI violations.""",
    
    "discard_variability": """Cluster {cluster_id} has too much waveform variability (amplitude CV={amplitude_cv:.2f}) 
for too few spikes ({n_spikes}). It is unlikely that any valid unit is embedded within this cluster. 
Additionally, it does not resemble other clusters sufficiently to be merged.""",
    
    "discard_isi": """Cluster {cluster_id} has far too high ISI violations ({isi_violation_rate:.1%}) for the number 
of spikes it contains ({n_spikes}). Too many spikes occur within the 2 ms refractory period. Combined with 
{additional_issue}, this cluster is not a valid single unit.""",
    
    "discard_isolation": """Cluster {cluster_id} has too few spikes ({n_spikes}) and is not similar enough to any 
other cluster's waveforms (max correlation={max_similarity:.2f} with cluster {most_similar_id}). Cannot be merged 
into any valid unit and has insufficient spike count to stand alone.""",
    
    "merge": """Cluster {cluster_id} and {merge_target} share sufficient waveform similarities in shape and amplitude 
(correlation={waveform_correlation:.2f}). Combined ISI violation rate is {post_merge_isi:.1%}, compared to 
{cluster_isi:.1%} and {target_isi:.1%} individually. The increase is negligible. Merging improves spike count 
({combined_n_spikes} total) while preserving unit quality.""",
    
    "keep": """Cluster {cluster_id} represents a valid single unit cluster. Waveform variability is low 
(amplitude CV={amplitude_cv:.2f}), ISI violation rate is acceptable ({isi_violation_rate:.1%}), and the waveform 
shape is consistent with physiological expectations (peak-to-trough={peak_to_trough:.2f} ms). The cluster contains 
{n_spikes} spikes, sufficient for reliable characterization."""
}


def build_phase1_prompt(cluster_id: int, n_spikes: int, n_overclusters: int) -> str:
    """Build Phase 1 VLM prompt with cluster context."""
    return f"""

## STEP 1: Neuronal Shape Check
First check if the waveform shape is neuronal:

Neuronal Shape Criteria:
{NEURONAL_CRITERIA}

If waveforms do NOT have valid neuronal shape → DISCARD.

## STEP 2: Split Decision (only if neuronal)
If waveforms ARE neuronal, check if cluster needs splitting:

Split Criteria (check waveform overlay + ISI histogram):
1. **Waveform Variability:** Do you see multiple distinct waveform families/shapes?
2. **Temporal Consistency:** Does the vertical spread (width) change significantly over time?
3. **ISI Pattern:** Does the ISI histogram suggest multiple units (high violations > {QUALITY_THRESHOLDS['max_isi_violation_rate']}, bimodal distribution)?
4. **Aggregation Complexity:** Is this cluster composed of many complex subclusters?

Decision Logic:
- NOT neuronal shape → DISCARD
- Neuronal + high variability/drift/ISI violations → SPLIT (identify tight subgroups)
- Neuronal + low variability + clean ISI → KEEP

You are judging Cluster {cluster_id} for spike sorting curation.

Cluster Summary:
- Spike count: {n_spikes}
- Composed of {n_overclusters} overclusters (hierarchical subcomponents)

Output only in JSON schema:
{{
  "action": "KEEP" | "DISCARD" | "SPLIT",
  "rationale": "Brief explanation (2-3 sentences)",
}}
Do not add any other text.
"""

def build_phase2_prompt(
    small_cluster_id: int,
    n_small: int,
    small_isi_rate: float,
    large_cluster_id: int,
    n_large: int,
    large_isi_rate: float,
    correlation: float,
    merged_isi_rate: float,
) -> str:
    """Build Phase 2 VLM prompt with merge context."""
    return f"""
## Decision Logic:

**STEP 1: Check small cluster quality**
- If small cluster (n < 1000) has excessive variability (too broad/noisy) → DISCARD
- If small cluster shape is not neuronal → DISCARD

**STEP 2: Check similarity to large cluster (only if small cluster is valid)**
- If waveforms looks similar AND merged ISI acceptable (< {QUALITY_THRESHOLDS['max_isi_violation_rate']}) → MERGE
- If waveforms are dissimilar → Choose "NOT_MERGE"

**STEP 3: Final decision**
- If you choose "NOT_MERGE" for ALL large clusters → small cluster will be DISCARDED
- So "NOT_MERGE" means: "These are different units, try other large clusters"
- Only use actual "DISCARD" if small cluster itself is invalid (bad shape/too noisy)

Neuronal Shape Criteria:
{NEURONAL_CRITERIA}

You are deciding whether to MERGE small cluster {small_cluster_id} into large cluster {large_cluster_id}.

Small Cluster {small_cluster_id} (post-split result):
- Spike count: {n_small}
- ISI violation rate: {small_isi_rate:.2%}

Large Cluster {large_cluster_id} (independent valid unit):
- Spike count: {n_large} (large, standalone cluster)
- ISI violation rate: {large_isi_rate:.2%}

Merge Prediction:
- Waveform correlation: {correlation:.3f}
- Merged ISI violation rate: {merged_isi_rate:.2%}
- Total spikes after merge: {n_small + n_large}

Output only in JSON schema:
{{
  "action": "MERGE" | "NOT_MERGE" | "DISCARD",
  "rationale": "Brief explanation (2-3 sentences)"
}}
Do not add any other text.
"""
