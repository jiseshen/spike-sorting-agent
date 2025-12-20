"""
Ablation test: Agent context WITHOUT quantitative metrics in prompts.

This module removes all numerical metrics from prompts to test whether VLM
can make accurate curation decisions based purely on visual information.

Removed from prompts:
- ISI violation rates (numerical values)
- Spike counts (numerical values)
- Waveform correlation scores
- Amplitude CV values
- All other quantitative metrics

Kept in prompts:
- Images (waveform overlays, ISI histograms, aggregation trees)
- Qualitative decision criteria
- Neuronal shape requirements
"""

# =====================================================================
# NEURONAL CRITERIA - Same as original (qualitative only)
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
Discard huge clusters (> 10000 spikes) with caution.
"""

# =====================================================================
# ABLATION PROMPTS - Visual-only decision making
# =====================================================================

def build_phase1_prompt_ablation(cluster_id: int, n_spikes: int, n_overclusters: int) -> str:
    """Build Phase 1 VLM prompt WITHOUT quality metrics.
    
    Removes:
    - ISI violation rate values (numerical)
    - ISI violation rate thresholds (e.g., "> 0.6%")
    - Amplitude CV values
    
    Keeps:
    - Spike counts (cluster size information)
    - Overcluster counts (structural complexity)
    - Images (waveform overlay, ISI histogram, aggregation tree)
    - Qualitative criteria
    - Decision logic
    """
    return f"""

## STEP 1: Neuronal Shape Check
First check if the waveform shape is neuronal:

Neuronal Shape Criteria:
{NEURONAL_CRITERIA}

If waveforms do NOT have valid neuronal shape → DISCARD.

## STEP 2: Split Decision (only if neuronal)
If waveforms ARE neuronal, check if cluster needs splitting:

Split Criteria (check waveform overlay + ISI histogram):
1. **Waveform Variability:** Do you see multiple distinct waveform families/shapes in the overlay?
2. **Temporal Consistency:** Does the vertical spread (width) change significantly over time?
3. **ISI Pattern:** Does the ISI histogram suggest multiple units (high refractory period violations, bimodal distribution)?
4. **Aggregation Complexity:** Does the tree show many complex subclusters that might represent embedded units?

Decision Logic:
- NOT neuronal shape → DISCARD
- Neuronal + high variability/drift/ISI violations visible → SPLIT (identify tight subgroups)
- Neuronal + low variability + clean ISI pattern → KEEP

You are judging Cluster {cluster_id} for spike sorting curation.

Cluster Summary:
- Spike count: {n_spikes}
- Composed of {n_overclusters} overclusters (hierarchical subcomponents)

**Examine the provided images carefully:**
- Waveform overlay: Look for shape consistency, spread, multiple families
- ISI histogram: Look for refractory violations (spikes < 2ms), distribution pattern
- Aggregation tree: Look for hierarchical structure complexity

Output only in JSON schema:
{{
  "action": "KEEP" | "DISCARD" | "SPLIT",
  "rationale": "Brief explanation (2-3 sentences) based on visual observations",
}}
Do not add any other text.
"""


def build_phase2_prompt_ablation(
    small_cluster_id: int,
    n_small: int,
    large_cluster_id: int,
    n_large: int,
) -> str:
    """Build Phase 2 VLM prompt WITHOUT quality metrics.
    
    Removes:
    - ISI violation rates (numerical values)
    - Waveform correlation scores (numerical values)
    - ISI violation rate thresholds (e.g., "< 0.6%")
    
    Keeps:
    - Spike counts (small cluster size, large cluster size)
    - Total spikes after merge
    - Images (small waveforms, large waveforms, merged ISI)
    - Qualitative criteria
    - Decision logic
    """
    return f"""
## Decision Logic:

**STEP 1: Check small cluster quality**
- If small cluster (n < 1000) has excessive variability (too broad/noisy) → DISCARD
- If small cluster shape is not neuronal → DISCARD

**STEP 2: Check similarity to large cluster (only if small cluster is valid)**
- If waveforms look similar in shape AND merged ISI histogram shows acceptable refractory violations → MERGE
- If waveforms are clearly dissimilar → Choose "NOT_MERGE"

**STEP 3: Final decision**
- If you choose "NOT_MERGE" for ALL large clusters → small cluster will be DISCARDED
- So "NOT_MERGE" means: "These are different units, try other large clusters"
- Only use actual "DISCARD" if small cluster itself is invalid (bad shape/too noisy)

Neuronal Shape Criteria:
{NEURONAL_CRITERIA}

You are deciding whether to MERGE small cluster {small_cluster_id} into large cluster {large_cluster_id}.

Small Cluster {small_cluster_id} (post-split result):
- Spike count: {n_small}

Large Cluster {large_cluster_id} (independent valid unit):
- Spike count: {n_large} (large, standalone cluster)

Merge Prediction:
- Total spikes after merge: {n_small + n_large}

**Examine the provided images carefully:**
- Small cluster waveforms: Check if valid neuronal shape, consistency
- Large cluster waveforms: Compare shape, amplitude, temporal profile
- Merged ISI histogram: Check if refractory violations remain acceptable after merge

Output only in JSON schema:
{{
  "action": "MERGE" | "NOT_MERGE" | "DISCARD",
  "rationale": "Brief explanation (2-3 sentences) based on visual observations"
}}
Do not add any other text.
"""


# =====================================================================
# QUALITY THRESHOLDS - Keep for internal logic (not in prompts)
# =====================================================================

QUALITY_THRESHOLDS = {
    # Spike count
    "auto_discard_threshold": 500,
    "min_spikes_standalone": 500,
    "min_spikes_merge": 500,
    
    # ISI violations
    "max_isi_violation_rate": 0.006,
    
    # Waveform consistency
    "max_amplitude_cv": 0.3,
    "max_amplitude_cv_large": 0.5,
    
    # Similarity
    "min_similarity_merge": 0.8,
    "min_similarity_hierarchy": 0.3,
    
    # Waveform shape
    "max_peak_to_trough_ms": 0.6,
    "min_peak_to_trough_ms": 0.15,
    "max_phases": 3,
    "min_phases": 2,
    
    # Temporal drift
    "max_width_change_pct": 0.2,
    
    # Cluster size classification
    "large_cluster_threshold": 500,
    "split_priority_threshold": 5000,
    "final_minimum_threshold": 5000,
}
