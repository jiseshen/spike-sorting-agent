"""
VLM runner for visual cluster analysis.

Packages waveform visuals, metadata, and prompts for VLM API calls.
Parses JSON decisions and interfaces with ClusterManager.
"""

import json
import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .agent_context import (
    NEURONAL_CRITERIA,
    build_phase1_prompt,
    build_phase2_prompt,
)

# Import VLM API
try:
    from .vlm_api import call_vlm
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    print("[Warning] VLM API not available, using mock responses")


# =====================================================================
# IMAGE UTILITIES
# =====================================================================

_VLM_CALL_INDEX = 0  # Monotonic counter for unique logging filenames per process

def _save_vlm_inputs(
    output_dir: Optional[Path],
    prefix: str,
    images: List[str],
    prompt: str,
    image_names: List[str],
    extra_meta: Optional[Dict[str, Any]] = None,
):
    """
    Save VLM input images and prompt to output directory with a UNIQUE call suffix.

    Each invocation gets a monotonically increasing call index to avoid overwrites when
    the same cluster is revisited (e.g. iterative splits). A CSV log is appended tracking
    all saved artifacts.

    Args:
        output_dir: Base output directory
        prefix: Base filename prefix (e.g., "phase1_cluster_123")
        images: List of base64-encoded images
        prompt: Full text prompt
        image_names: Names for each image (e.g., ["waveform", "isi", "tree"])
        extra_meta: Optional dict with additional metadata (e.g., phase, cluster ids)
    """
    if output_dir is None:
        return

    global _VLM_CALL_INDEX
    _VLM_CALL_INDEX += 1
    call_id = _VLM_CALL_INDEX
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_prefix = f"{prefix}_call{call_id:05d}"

    # Create vlm_inputs subdirectory
    vlm_dir = output_dir / "vlm_inputs"
    vlm_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt
    prompt_file = vlm_dir / f"{unique_prefix}_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)

    saved_image_files: List[str] = []
    # Save images
    for img_b64, img_name in zip(images, image_names):
        img_file = vlm_dir / f"{unique_prefix}_{img_name}.png"
        img_data = base64.b64decode(img_b64)
        with open(img_file, 'wb') as f:
            f.write(img_data)
        saved_image_files.append(str(img_file.name))

    # Append to CSV log (create header if not exists)
    log_path = vlm_dir / "vlm_call_log.csv"
    header_needed = not log_path.exists()
    row = {
        "call_id": call_id,
        "timestamp": timestamp,
        "base_prefix": prefix,
        "unique_prefix": unique_prefix,
        "prompt_file": prompt_file.name,
        "image_files": ";".join(saved_image_files),
    }
    if extra_meta:
        # Flatten extra_meta keys (only simple values)
        for k, v in extra_meta.items():
            row[f"meta_{k}"] = v

    # Ensure consistent column ordering when appending
    import csv as _csv
    if header_needed:
        with open(log_path, 'w', newline='') as lf:
            writer = _csv.writer(lf)
            writer.writerow(list(row.keys()))
            writer.writerow(list(row.values()))
    else:
        # Read existing header to preserve column order; extend if new meta keys appear
        with open(log_path, 'r', newline='') as lf:
            reader = _csv.reader(lf)
            existing_rows = list(reader)
        existing_header = existing_rows[0]
        new_cols = [c for c in row.keys() if c not in existing_header]
        if new_cols:
            existing_header.extend(new_cols)
            # Re-write file with extended header
            with open(log_path, 'w', newline='') as lf:
                writer = _csv.writer(lf)
                writer.writerow(existing_header)
                for old_row in existing_rows[1:]:
                    # Pad old rows with blanks for new columns
                    old_row.extend(["" for _ in new_cols])
                    writer.writerow(old_row)
        # Append new row in header order
        with open(log_path, 'a', newline='') as lf:
            writer = _csv.writer(lf)
            writer.writerow([row.get(col, "") for col in existing_header])



def _sanitize_json_response(text: str) -> str:
    """Sanitize model output into strict JSON.

    Handles common patterns:
    - Code fences like ```json ... ```
    - Leading text before/after JSON
    - Inline // comments
    - Trailing commas before ] or }
    - Newlines and extra whitespace

    Returns a JSON string starting with '{' and ending with the matching '}'.
    Raises ValueError if no JSON object can be extracted.
    """
    import re

    # Strip code fences
    text = re.sub(r"```(?:json)?\n?|```", "", text, flags=re.IGNORECASE)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Find the first '{' and extract a balanced JSON object
    start = text.find('{')
    if start == -1:
        raise ValueError('No JSON object start found')

    # Scan to find the matching closing brace accounting for braces inside strings
    brace_count = 0
    in_string = False
    escape = False
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
    if end is None:
        raise ValueError('No balanced JSON object found')

    json_str = text[start:end+1]

    # Remove // comments
    json_str = re.sub(r"//.*", "", json_str)

    # Remove trailing commas before ] or }
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)

    return json_str

def fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def create_waveform_overlay_image(
    waveforms: np.ndarray,
    cluster_id: int,
    sampling_rate: float = 30000.0,
    max_waveforms: int = 5000,
) -> str:
    """
    Create waveform overlay visualization.
    
    Args:
        waveforms: (n_spikes, n_samples) array
        cluster_id: Cluster identifier
        sampling_rate: Hz
        max_waveforms: Maximum number to plot
        
    Returns:
        Base64-encoded PNG image
    """
    n_spikes, n_samples = waveforms.shape
    time_ms = np.arange(n_samples) / sampling_rate * 1000
    
    # Subsample to max_waveforms for plotting if needed
    if n_spikes > max_waveforms:
        indices = np.random.choice(n_spikes, max_waveforms, replace=False)
        plot_waveforms = waveforms[indices]
    else:
        plot_waveforms = waveforms

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot individual waveforms with transparency (no median/mean)
    for wf in plot_waveforms:
        ax.plot(time_ms, wf, 'steelblue', alpha=0.3, linewidth=0.8)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (μV)')
    ax.set_title(f'Cluster {cluster_id} Waveform Overlay (n={n_spikes})')
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def create_isi_histogram_image(
    spike_times: np.ndarray,
    cluster_id: int,
    max_isi_ms: float = 50.0,
) -> str:
    """
    Create ISI histogram visualization.
    
    Args:
        spike_times: Spike times in seconds
        cluster_id: Cluster identifier
        max_isi_ms: Maximum ISI to display (ms)
        
    Returns:
        Base64-encoded PNG image
    """
    if len(spike_times) < 2:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f'Cluster {cluster_id}: < 2 spikes', 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig_to_base64(fig)
    
    # Compute ISIs
    isis = np.diff(np.sort(spike_times)) * 1000  # Convert to ms
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Histogram
    bins = np.arange(0, max_isi_ms, 0.5)
    ax.hist(isis[isis < max_isi_ms], bins=bins, color='steelblue', edgecolor='black')
    
    # Mark refractory period
    ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='2 ms (refractory)')
    
    # Compute violation rate
    violations = np.sum(isis < 2.0)
    violation_rate = violations / len(isis) if len(isis) > 0 else 0
    
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('Count')
    ax.set_title(f'Cluster {cluster_id} ISI Histogram (violations: {violation_rate:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def create_amplitude_distribution_image(
    waveforms: np.ndarray,
    cluster_id: int,
) -> str:
    """
    Create amplitude distribution visualization.
    
    Args:
        waveforms: (n_spikes, n_samples) array
        cluster_id: Cluster identifier
        
    Returns:
        Base64-encoded PNG image
    """
    # Compute peak-to-trough amplitudes
    peak_to_trough = np.max(waveforms, axis=1) - np.min(waveforms, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.hist(peak_to_trough, bins=50, color='steelblue', edgecolor='black')
    
    mean_amp = np.mean(peak_to_trough)
    cv = np.std(peak_to_trough) / mean_amp if mean_amp > 0 else 0
    
    ax.axvline(mean_amp, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_amp:.1f} μV')
    
    ax.set_xlabel('Peak-to-Trough Amplitude (μV)')
    ax.set_ylabel('Count')
    ax.set_title(f'Cluster {cluster_id} Amplitude Distribution (CV={cv:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def create_aggregation_tree_image(
    hierarchy_tree: np.ndarray,
    overcluster_composition: List[int],
    cluster_id: int,
) -> str:
    """
    Create aggregation tree visualization matching MATLAB aggtree.m style.
    
    Shows hierarchical merge structure with bracket-style connections:
    - Leaf nodes (overclusters) at bottom
    - Brackets connecting merges, height = dissimilarity
    - Color coding by merge chronology
    
    Args:
        hierarchy_tree: (4, n_merges) tree [src1, src2, similarity, unused]
                       Format: src2 merged INTO src1
        overcluster_composition: List of overcluster IDs in this cluster
        cluster_id: Cluster identifier
        
    Returns:
        Base64-encoded PNG image
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_overclusters = len(overcluster_composition)
    
    # Handle trivial case
    if n_overclusters == 1:
        ax.text(0.5, 0.5, f'Cluster {cluster_id}: Single Overcluster\n(No hierarchical structure)',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
        ax.axis('off')
        return fig_to_base64(fig)
    
    # Extract relevant merges involving our overclusters
    oc_set = set(overcluster_composition)
    
    # CRITICAL: hierarchy_tree merges work in-place!
    # When (src1, src2, ...) is executed, src2 is merged INTO src1,
    # and future references to src1 include all content from src2.
    # We need to track the actual merge structure by simulating the sequential merges.
    
    # Track which overclusters each node currently contains (changes as merges happen)
    node_contents = {oc: {oc} for oc in oc_set}  # Initialize with original overclusters
    
    # Track the merge history: which merges involved our overclusters
    relevant_merges = []
    
    # Process merges in chronological order
    for step_idx in range(hierarchy_tree.shape[1]):
        src1 = int(hierarchy_tree[0, step_idx])
        src2 = int(hierarchy_tree[1, step_idx])
        similarity = hierarchy_tree[2, step_idx]
        dissimilarity = 1.0 - similarity
        
        # Check if this merge involves our overclusters
        has_src1 = src1 in node_contents and node_contents[src1] & oc_set
        has_src2 = src2 in node_contents and node_contents[src2] & oc_set
        
        if not has_src1 and not has_src2:
            # Neither node contains our overclusters, skip
            continue
        
        if has_src1 and has_src2:
            # Both nodes contain our overclusters - this is an internal merge
            relevant_merges.append({
                'parent': src1,  # src1 becomes the parent (absorbs src2)
                'lchild': src1,  # src1's current content
                'rchild': src2,  # src2's current content
                'dissimilarity': dissimilarity,
                'step': step_idx,
            })
            
            # Update: src1 now contains everything from both src1 and src2
            contents1 = node_contents.get(src1, set())
            contents2 = node_contents.get(src2, set())
            node_contents[src1] = contents1 | contents2
            
            # src2 no longer exists as a standalone node (absorbed into src1)
            if src2 in node_contents:
                del node_contents[src2]
        else:
            # One of them contains our overclusters, the other doesn't
            # Still need to track this because the node gets absorbed into the other
            if has_src1:
                # src1 has our clusters, src2 doesn't - src2 gets absorbed but we don't care
                # Update src1 to include src2's content (even though src2 isn't in our set)
                if src2 in node_contents:
                    contents2 = node_contents[src2]
                    node_contents[src1] = node_contents[src1] | contents2
                    del node_contents[src2]
            else:
                # src2 has our clusters, src1 doesn't - src2 gets absorbed into src1
                # Now src1 contains our overclusters!
                if src2 in node_contents:
                    contents2 = node_contents[src2]
                    if src1 in node_contents:
                        node_contents[src1] = node_contents[src1] | contents2
                    else:
                        node_contents[src1] = contents2.copy()
                    del node_contents[src2]
    
    # After processing all merges, node_contents contains the final state
    # Find which node(s) contain our overclusters
    final_roots = []
    for node_id, contents in node_contents.items():
        if contents & oc_set:  # Contains at least some of our overclusters
            final_roots.append((node_id, contents))
    
    # Check if all overclusters are unified into a single root
    complete_roots = [(nid, cont) for nid, cont in final_roots if cont == oc_set]
    
    if not relevant_merges:
        # No hierarchical structure - show flat
        x_positions = np.arange(n_overclusters) + 1
        ax.bar(x_positions, [0.1] * n_overclusters,
               color='steelblue', alpha=0.7, edgecolor='black', width=0.6)
        ax.set_xlabel('Overcluster ID', fontsize=11)
        ax.set_ylabel('Merge Height', fontsize=11)
        ax.set_title(f'Cluster {cluster_id}: Flat Structure\n{n_overclusters} overclusters (no internal merges)',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(oc) for oc in sorted(overcluster_composition)], rotation=45)
        plt.tight_layout()
        return fig_to_base64(fig)
    
    # Now build the tree structure for visualization
    # We need to track which nodes are leaves vs internal nodes
    # Problem: In the merge list, 'lchild' and 'rchild' might themselves be results of previous merges
    
    # Build a proper tree by tracking the hierarchy
    # For each merge: parent absorbs both lchild and rchild
    # But lchild at time of merge might be the result of a previous merge
    
    # Strategy: Build tree bottom-up
    # 1. Identify all leaf nodes (original overclusters that appear in the tree)
    # 2. For each merge, track parent-child relationships
    
    # Collect all nodes involved
    all_nodes = set(oc_set)  # Start with original overclusters
    for merge in relevant_merges:
        all_nodes.add(merge['parent'])
        # Note: lchild and rchild at merge time might be intermediate nodes
        # But we've already recorded them correctly in the merge
    
    # Build node_info with parent-child relationships
    node_info = {nid: {'parent': None, 'children': [], 'xpos': 0, 'ypos': 0, 'is_leaf': nid in oc_set} 
                 for nid in all_nodes}
    
    # Process merges to build tree structure
    # Key insight: we need to track the state of each node at the time of each merge
    # When merge happens: parent already has some content (lchild), absorbs rchild
    
    # Actually, let's think differently:
    # Each merge creates a new "version" of src1 that includes src2
    # But for visualization, we want to show the hierarchical structure
    
    # Let me re-approach: create synthetic nodes for each merge to show the tree properly
    max_orig = max(oc_set) if oc_set else 0
    synthetic_node_id = max_orig + 1
    
    # Map from (node_id, step) -> synthetic_node for that state
    node_state_map = {}  # Maps original node to its current synthetic representation
    
    # Initialize: each overcluster starts as itself
    for oc in oc_set:
        node_state_map[oc] = oc
    
    synthetic_merges = []
    for merge in relevant_merges:
        parent_orig = merge['parent']
        lchild_orig = merge['lchild']  # This is parent_orig before merge
        rchild_orig = merge['rchild']
        
        # Get current synthetic nodes for lchild and rchild
        lchild_synthetic = node_state_map.get(lchild_orig, lchild_orig)
        rchild_synthetic = node_state_map.get(rchild_orig, rchild_orig)
        
        # Create synthetic node for the merged result
        parent_synthetic = synthetic_node_id
        synthetic_node_id += 1
        
        synthetic_merges.append({
            'parent': parent_synthetic,
            'lchild': lchild_synthetic,
            'rchild': rchild_synthetic,
            'dissimilarity': merge['dissimilarity'],
            'step': merge['step'],
        })
        
        # Update mapping: parent_orig now maps to the new synthetic node
        node_state_map[parent_orig] = parent_synthetic
    
    # Build tree structure with synthetic nodes
    all_nodes = set(oc_set)
    for merge in synthetic_merges:
        all_nodes.add(merge['parent'])
        all_nodes.add(merge['lchild'])
        all_nodes.add(merge['rchild'])
    
    node_info = {nid: {'parent': 0, 'lchild': 0, 'rchild': 0, 'xpos': 0, 'ypos': 0} 
                 for nid in all_nodes}
    
    # Set parent/child relationships and y positions
    # MATLAB logic: ypos(parent) = max(ypos([lchild,rchild])) + atree(step,4)
    # where atree(:,4) is related to dissimilarity but we use it as height increment
    for merge in synthetic_merges:
        parent = merge['parent']
        lchild = merge['lchild']
        rchild = merge['rchild']
        dissim = merge['dissimilarity']
        
        node_info[parent]['lchild'] = lchild
        node_info[parent]['rchild'] = rchild
        node_info[lchild]['parent'] = parent
        node_info[rchild]['parent'] = parent
        
        # Y position: max of children + height based on dissimilarity
        # Use dissimilarity as relative height (so similar merges are lower)
        node_info[parent]['ypos'] = max(node_info[lchild]['ypos'], 
                                       node_info[rchild]['ypos']) + dissim
    
    # Find tree roots (nodes with no parent)
    tree_roots = [nid for nid in all_nodes if node_info[nid]['parent'] == 0]
    
    # Assign x positions using depth-first traversal (left to right)
    x_counter = [1]  # Use list to make it mutable in nested function
    
    def assign_x_positions(node_id):
        """Recursively assign x positions left-to-right"""
        if node_info[node_id]['lchild'] == 0:
            # Leaf node
            node_info[node_id]['xpos'] = x_counter[0]
            x_counter[0] += 1
        else:
            # Interior node - traverse children first
            assign_x_positions(node_info[node_id]['lchild'])
            assign_x_positions(node_info[node_id]['rchild'])
            # X position is average of children
            node_info[node_id]['xpos'] = (node_info[node_info[node_id]['lchild']]['xpos'] + 
                                         node_info[node_info[node_id]['rchild']]['xpos']) / 2.0
    
    for root in tree_roots:
        assign_x_positions(root)
    
    # Draw the tree
    # 1. Draw nodes
    for nid in all_nodes:
        x = node_info[nid]['xpos']
        y = node_info[nid]['ypos']
        if node_info[nid]['lchild'] == 0:
            # Leaf node - larger marker
            ax.plot(x, y, 'o', color='steelblue', markersize=8, markeredgecolor='black', markeredgewidth=1)
        else:
            # Interior node
            ax.plot(x, y, 'o', color='gray', markersize=6)
    
    # 2. Draw bracket connections with color coding by merge order
    cmap = plt.cm.winter
    n_merges = len(synthetic_merges)
    colors = [cmap(i / max(n_merges - 1, 1)) for i in range(n_merges)]
    
    for idx, merge in enumerate(synthetic_merges):
        parent = merge['parent']
        lchild = merge['lchild']
        rchild = merge['rchild']
        
        x_parent = node_info[parent]['xpos']
        y_parent = node_info[parent]['ypos']
        x_left = node_info[lchild]['xpos']
        y_left = node_info[lchild]['ypos']
        x_right = node_info[rchild]['xpos']
        y_right = node_info[rchild]['ypos']
        
        # Draw bracket: left vertical, horizontal, right vertical
        ax.plot([x_left, x_left], [y_left, y_parent], color=colors[idx], linewidth=2)
        ax.plot([x_left, x_right], [y_parent, y_parent], color=colors[idx], linewidth=2)
        ax.plot([x_right, x_right], [y_right, y_parent], color=colors[idx], linewidth=2)
    
    # 3. Draw final cluster markers at top
    max_y = max(node_info[nid]['ypos'] for nid in all_nodes)
    y_top = max_y * 1.15
    
    for root_idx, root in enumerate(tree_roots):
        x_root = node_info[root]['xpos']
        y_root = node_info[root]['ypos']
        
        # Line to top
        ax.plot([x_root, x_root], [y_root, y_top], 'k-', linewidth=2.5)
        ax.plot(x_root, y_top, 'ko', markersize=10, markerfacecolor='black')
        
        # Label
        stagger = (max_y * 0.05) * (root_idx % 3)
        ax.text(x_root, y_top * 1.05 + stagger, f'Root', 
               ha='center', fontsize=10, fontweight='bold')
    
    # 4. Add leaf labels at bottom
    leaf_nodes = [nid for nid in all_nodes if node_info[nid]['lchild'] == 0]
    leaf_labels = [str(nid) for nid in sorted(leaf_nodes, key=lambda n: node_info[n]['xpos'])]
    leaf_x_positions = sorted([node_info[nid]['xpos'] for nid in leaf_nodes])
    
    ax.set_xticks(leaf_x_positions)
    ax.set_xticklabels(leaf_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Overcluster ID (original leaf nodes)', fontsize=11)
    ax.set_ylabel('Merge Height (cumulative dissimilarity)', fontsize=11)
    
    # Title should reflect the structure
    n_roots = len(tree_roots)
    if n_roots == 1:
        title = f'Cluster {cluster_id}: Hierarchical Aggregation Tree\n{n_overclusters} overclusters, {n_merges} internal merges'
    else:
        title = f'Cluster {cluster_id}: Aggregation Forest ({n_roots} separate trees)\n{n_overclusters} overclusters, {n_merges} internal merges'
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(0, len(leaf_nodes) + 1)
    ax.set_ylim(-max_y * 0.05, y_top * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation
    n_roots = len(tree_roots)
    if n_merges == n_overclusters - 1:
        interp_text = f'Complete tree: All {n_overclusters} overclusters hierarchically merged into single root'
        interp_color = 'lightgreen'
    elif n_roots == 1:
        interp_text = f'Partial tree: {n_merges} merges ({n_overclusters - n_merges - 1} unmerged overclusters)'
        interp_color = 'lightyellow'
    elif n_merges > 0:
        interp_text = f'Forest structure: {n_roots} separate trees (overclusters weakly connected in original clustering)'
        interp_color = 'lightcoral'
    else:
        interp_text = f'Flat structure: {n_overclusters} overclusters (no hierarchical merges)'
        interp_color = 'lightcoral'
    
    ax.text(0.02, 0.98, interp_text,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.6))
    
    # Add colorbar to show merge chronology
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_merges-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Merge Order (blue=early, green=late)', 
                       orientation='horizontal', pad=0.15, aspect=30)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def create_waveform_comparison_image(
    small_waveforms: np.ndarray,
    large_waveforms: np.ndarray,
    small_spike_times: np.ndarray,
    large_spike_times: np.ndarray,
    small_cluster_id: int,
    large_cluster_id: int,
    sampling_rate: float = 30000.0,
) -> tuple:
    """
    Create 3 separate images for Phase 2 waveform comparison.
    
    Returns separate figures for small/large waveforms + merged ISI histogram.
    
    Args:
        small_waveforms: (n_spikes_small, n_samples)
        large_waveforms: (n_spikes_large, n_samples)
        small_spike_times: Spike times in seconds
        large_spike_times: Spike times in seconds
        small_cluster_id: Small cluster ID
        large_cluster_id: Large cluster ID
        sampling_rate: Hz
        
    Returns:
        tuple: (small_wf_img_b64, large_wf_img_b64, merged_isi_img_b64)
    """
    # Reuse overlay image utility for consistent visualization (cap at 5000)
    small_wf_img = create_waveform_overlay_image(
        waveforms=small_waveforms,
        cluster_id=small_cluster_id,
        sampling_rate=sampling_rate,
        max_waveforms=5000,
    )
    large_wf_img = create_waveform_overlay_image(
        waveforms=large_waveforms,
        cluster_id=large_cluster_id,
        sampling_rate=sampling_rate,
        max_waveforms=5000,
    )
    
    # Merged ISI histogram (hypothetical)
    merged_times = np.sort(np.concatenate([small_spike_times, large_spike_times]))
    merged_isi_hist = create_isi_histogram_image(merged_times, f"{small_cluster_id}+{large_cluster_id}")
    
    return small_wf_img, large_wf_img, merged_isi_hist


# =====================================================================
# VLM API INTERFACE (Mock for now - integrate with actual API)
# =====================================================================

def call_vlm_api(
    prompt: str,
    images: List[str],
    model: str = "gpt-4o",
    provider: str = "gpt4o",
    use_mock: bool = False,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """
    Call VLM API with prompt and images, with automatic retry on failure.
    
    Args:
        prompt: System/user prompt
        images: List of base64-encoded images
        model: Model identifier
            - Vision models: "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"
            - Reasoning models: "o1-preview", "o1-mini", "o3-mini"
            - Claude: "claude-3-5-sonnet-20241022"
        provider: API provider ("gpt4o" or "claude")
        use_mock: If True, return mock response (for testing)
        temperature: Sampling temperature (ignored for reasoning models)
        reasoning_effort: For reasoning models: "low", "medium", "high" (None = auto)
        max_retries: Maximum number of retry attempts on failure
        
    Returns:
        Raw JSON response string
    """
    # Use mock if requested or if API not available
    if use_mock or not VLM_AVAILABLE:
        print(f"[VLM Mock] Called {model} with {len(images)} images")
        print(f"[VLM Mock] Prompt preview: {prompt[:200]}...")
        
        # Mock response structure - use KEEP for Phase1 to allow pipeline to continue
        if "STEP 1: Neuronal Shape Check" in prompt or "Cluster Summary" in prompt:
            # Phase 1: Neuronal + Split check
            # Use KEEP to allow pipeline to continue testing
            mock_response = {
                "action": "KEEP",
                "rationale": "Mock VLM: Accepting cluster to test pipeline flow",
                "split_groups": []
            }
        else:
            # Phase 2: Merge decision
            mock_response = {
                "action": "NOT_MERGE",
                "rationale": "Mock VLM: Waveforms show different shapes, likely different units"
            }
        
        return json.dumps(mock_response, indent=2)
    
    # Real API call with retry
    for attempt in range(max_retries):
        try:
            # Detect reasoning model
            is_reasoning = any(rm in model for rm in ["o1", "o3"])
            
            if is_reasoning:
                effort_str = f" (effort={reasoning_effort})" if reasoning_effort else " (auto effort)"
                print(f"[VLM API] Calling {provider}/{model}{effort_str} with {len(images)} images...")
            else:
                print(f"[VLM API] Calling {provider}/{model} (temp={temperature}) with {len(images)} images...")
            
            raw_response = call_vlm(
                prompt=prompt,
                images=images,
                provider=provider,
                model=model,
                max_tokens=1000,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
            print(f"[VLM API] Response received ({len(raw_response)} chars)")
            return raw_response
        
        except Exception as e:
            print(f"[VLM API Error] Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                print(f"[VLM API] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Final attempt failed, fall back to mock
                print(f"[VLM API] All {max_retries} attempts failed, falling back to mock response")
                return call_vlm_api(prompt, images, model, provider, use_mock=True)


# =====================================================================
# PHASE 1: VLM ITERATIVE SPLIT/DISCARD
# =====================================================================

def vlm_neuronal_validity_check(
    cluster_id: int,
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    sampling_rate: float = 30000.0,
) -> Dict[str, Any]:
    """
    VLM judges ONLY neuronal validity (for baseline pipeline).
    
    Returns only KEEP or DISCARD (no SPLIT option).
    Used in baseline where splits are done heuristically.
    
    Args:
        cluster_id: Target cluster ID
        waveforms: (n_spikes, n_samples) array
        spike_times: Spike times in seconds
        sampling_rate: Hz
        
    Returns:
        Dict with:
            - action: "KEEP" | "DISCARD"
            - rationale: Reasoning string
            - raw_response: Original VLM response
    """
    from .agent_context import NEURONAL_CRITERIA
    
    # Create only waveform overlay (no tree needed for simple neuronal check)
    wf_overlay = create_waveform_overlay_image(waveforms, cluster_id, sampling_rate)
    isi_hist = create_isi_histogram_image(spike_times, cluster_id)
    
    # Simple neuronal check prompt
    prompt = f"""
You are checking if Cluster {cluster_id} has neuronal waveform shape.

Cluster Summary:
- Spike count: {len(spike_times)}

## Task: Neuronal Shape Validation

Check if the waveform shape matches valid neuronal action potentials:

{NEURONAL_CRITERIA}

**Decision:**
- If waveforms have valid neuronal shape → KEEP
- If waveforms do NOT have neuronal shape (artifact, noise, non-physiological) → DISCARD

Output JSON:
{{
  "action": "KEEP" | "DISCARD",
  "rationale": "Brief explanation (1-2 sentences)"
}}

Focus ONLY on neuronal shape validity. Do not consider clustering quality or splitting.
"""
    
    # Call VLM with retry on parse failure (max 3 attempts)
    max_parse_attempts = 3
    for parse_attempt in range(max_parse_attempts):
        raw_response = call_vlm_api(
            prompt=prompt,
            images=[wf_overlay, isi_hist],
        )
        
        # Parse JSON (robust to code fences/comments/trailing commas)
        try:
            sanitized = _sanitize_json_response(raw_response)
            decision = json.loads(sanitized)
            action = decision.get("action", "DISCARD")
            if action not in ["KEEP", "DISCARD"]:
                action = "DISCARD"
            return {
                "action": action,
                "rationale": decision.get("rationale", ""),
                "raw_response": raw_response,
            }
        except Exception as e:
            print(f"[VLM Parse Error] Attempt {parse_attempt + 1}/{max_parse_attempts}: {e}")
            print(f"[VLM Raw Response] {raw_response[:500]}...")
            
            if parse_attempt < max_parse_attempts - 1:
                print(f"[VLM] Retrying VLM call due to parse failure...")
                time.sleep(1)  # Brief pause before retry
            else:
                # All retries exhausted, fall back to default
                print(f"[VLM] All {max_parse_attempts} parse attempts failed, falling back to DISCARD")
                return {
                    "action": "DISCARD",
                    "rationale": f"JSON parse error after {max_parse_attempts} attempts",
                    "raw_response": raw_response,
                }


def vlm_phase1_cluster_decision(
    cluster_id: int,
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    overcluster_composition: List[int],
    hierarchy_tree: np.ndarray,
    sampling_rate: float = 30000.0,
    provider: str = "gpt4o",
    model: str = "gpt-4o",
    use_mock: bool = False,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    VLM judges a single cluster: neuronal validity + split/discard decision.
    
    Iteratively applied until VLM says cluster is clean (no more splits needed).
    
    Args:
        cluster_id: Target cluster ID
        waveforms: (n_spikes, n_samples) array
        spike_times: Spike times in seconds
        overcluster_composition: List of overcluster IDs in this cluster
        hierarchy_tree: (4, n_merges) aggregation tree
        sampling_rate: Hz
        provider: VLM provider ("gpt4o", "claude")
        model: Model name
        use_mock: Use mock responses for testing
        temperature: Sampling temperature (vision models only)
        reasoning_effort: Reasoning effort level (reasoning models only)
        output_dir: Directory to save VLM inputs (images + prompts)
        
    Returns:
        Dict with:
            - action: "KEEP" | "DISCARD" | "SPLIT"
            - rationale: Reasoning string
            - split_groups: List[List[int]] overcluster IDs for each subgroup (if action="SPLIT")
            - raw_response: Original VLM response
    """
    # Create visualizations
    wf_overlay = create_waveform_overlay_image(waveforms, cluster_id, sampling_rate)
    isi_hist = create_isi_histogram_image(spike_times, cluster_id)
    tree_viz = create_aggregation_tree_image(hierarchy_tree, overcluster_composition, cluster_id)
    
    # Build prompt with context
    prompt = build_phase1_prompt(
        cluster_id=cluster_id,
        n_spikes=len(spike_times),
        n_overclusters=len(overcluster_composition),
    )
    
    # Save inputs if output_dir provided
    _save_vlm_inputs(
        output_dir=output_dir,
        prefix=f"phase1_cluster_{cluster_id}",
        images=[wf_overlay, isi_hist, tree_viz],
        prompt=prompt,
        image_names=["waveform", "isi", "tree"],
    )
    
    # Call VLM with retry on parse failure (max 3 attempts)
    max_parse_attempts = 3
    for parse_attempt in range(max_parse_attempts):
        raw_response = call_vlm_api(
            prompt=prompt,
            images=[wf_overlay, isi_hist, tree_viz],
            model=model,
            provider=provider,
            use_mock=use_mock,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        
        # Parse JSON (robust) — only action + rationale needed
        try:
            sanitized = _sanitize_json_response(raw_response)
            decision = json.loads(sanitized)
            action = decision.get("action", "DISCARD")
            return {
                "action": action,
                "rationale": decision.get("rationale", ""),
                "raw_response": raw_response,
            }
        except Exception as e:
            print(f"[VLM Parse Error] Attempt {parse_attempt + 1}/{max_parse_attempts}: {e}")
            print(f"[VLM Raw Response] {raw_response[:500]}...")
            
            if parse_attempt < max_parse_attempts - 1:
                print(f"[VLM] Retrying VLM call due to parse failure...")
                time.sleep(1)  # Brief pause before retry
            else:
                # All retries exhausted, fall back to default
                print(f"[VLM] All {max_parse_attempts} parse attempts failed, falling back to DISCARD")
                return {
                    "action": "DISCARD",
                    "rationale": f"JSON parse error after {max_parse_attempts} attempts",
                    "raw_response": raw_response,
                }


# =====================================================================
# PHASE 2: VLM MERGE/DISCARD DECISIONS
# =====================================================================

def vlm_phase2_merge_decision(
    small_cluster_id: int,
    small_waveforms: np.ndarray,
    small_spike_times: np.ndarray,
    large_cluster_id: int,
    large_waveforms: np.ndarray,
    large_spike_times: np.ndarray,
    sampling_rate: float = 30000.0,
    provider: str = "gpt4o",
    model: str = "gpt-4o",
    use_mock: bool = False,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    VLM judges whether to merge small cluster into large cluster.
    
    Shows separate waveforms + predicted merged ISI/histogram.
    
    Args:
        small_cluster_id: Small cluster ID
        small_waveforms: (n_spikes_small, n_samples)
        small_spike_times: Spike times in seconds
        large_cluster_id: Large cluster ID
        large_waveforms: (n_spikes_large, n_samples)
        large_spike_times: Spike times in seconds
        sampling_rate: Hz
        provider: VLM provider ("gpt4o", "claude")
        model: Model name
        use_mock: Use mock responses for testing
        temperature: Sampling temperature (vision models only)
        reasoning_effort: Reasoning effort level (reasoning models only)
        output_dir: Directory to save VLM inputs (images + prompts)
        
    Returns:
        Dict with:
            - action: "MERGE" | "NOT_MERGE" | "DISCARD"
            - rationale: Reasoning string
            - raw_response: Original VLM response
    """
    # Use overlay image utility for consistent visualization (cap at 5000)
    small_wf_img = create_waveform_overlay_image(
        waveforms=small_waveforms,
        cluster_id=small_cluster_id,
        sampling_rate=sampling_rate,
        max_waveforms=5000,
    )
    large_wf_img = create_waveform_overlay_image(
        waveforms=large_waveforms,
        cluster_id=large_cluster_id,
        sampling_rate=sampling_rate,
        max_waveforms=5000,
    )
    
    # Merged ISI histogram (hypothetical)
    merged_times = np.sort(np.concatenate([small_spike_times, large_spike_times]))
    merged_isi_hist = create_isi_histogram_image(merged_times, f"{small_cluster_id}+{large_cluster_id}")
    
    # Compute correlation and ISI stats
    corr = compute_waveform_correlation(small_waveforms, large_waveforms)
    merged_isi_rate = compute_merged_isi_violation_rate(small_spike_times, large_spike_times)
    small_isi_rate = compute_merged_isi_violation_rate(small_spike_times, small_spike_times[:1])  # Self
    large_isi_rate = compute_merged_isi_violation_rate(large_spike_times, large_spike_times[:1])  # Self
    
    # Build prompt
    prompt = build_phase2_prompt(
        small_cluster_id=small_cluster_id,
        n_small=len(small_spike_times),
        small_isi_rate=small_isi_rate,
        large_cluster_id=large_cluster_id,
        n_large=len(large_spike_times),
        large_isi_rate=large_isi_rate,
        correlation=corr,
        merged_isi_rate=merged_isi_rate,
    )
    
    # Save inputs if output_dir provided
    _save_vlm_inputs(
        output_dir=output_dir,
        prefix=f"phase2_merge_{small_cluster_id}_into_{large_cluster_id}",
        images=[small_wf_img, large_wf_img, merged_isi_hist],
        prompt=prompt,
        image_names=["small_waveform", "large_waveform", "merged_isi"],
    )
    
    # Call VLM with retry on parse failure (max 3 attempts)
    max_parse_attempts = 3
    for parse_attempt in range(max_parse_attempts):
        raw_response = call_vlm_api(
            prompt=prompt,
            images=[small_wf_img, large_wf_img, merged_isi_hist],
            model=model,
            provider=provider,
            use_mock=use_mock,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        
        # Parse JSON (robust)
        try:
            sanitized = _sanitize_json_response(raw_response)
            decision = json.loads(sanitized)
            return {
                "action": decision.get("action", "NOT_MERGE"),
                "rationale": decision.get("rationale", ""),
                "raw_response": raw_response,
            }
        except Exception as e:
            print(f"[VLM Parse Error] Attempt {parse_attempt + 1}/{max_parse_attempts}: {e}")
            print(f"[VLM Raw Response] {raw_response[:500]}...")
            
            if parse_attempt < max_parse_attempts - 1:
                print(f"[VLM] Retrying VLM call due to parse failure...")
                time.sleep(1)  # Brief pause before retry
            else:
                # All retries exhausted, fall back to default
                print(f"[VLM] All {max_parse_attempts} parse attempts failed, falling back to NOT_MERGE")
                return {
                    "action": "NOT_MERGE",
                    "rationale": f"JSON parse error after {max_parse_attempts} attempts",
                    "raw_response": raw_response,
                }


# =====================================================================
# UTILITIES
# =====================================================================

def compute_waveform_correlation(
    waveforms_a: np.ndarray,
    waveforms_b: np.ndarray,
    max_samples: int = 200,
) -> float:
    """
    Compute correlation between two clusters' median waveforms.
    
    Args:
        waveforms_a: (n_spikes_a, n_samples)
        waveforms_b: (n_spikes_b, n_samples)
        max_samples: Subsample limit
        
    Returns:
        Pearson correlation coefficient
    """
    # Subsample if needed
    if waveforms_a.shape[0] > max_samples:
        indices = np.random.choice(waveforms_a.shape[0], max_samples, replace=False)
        waveforms_a = waveforms_a[indices]
    
    if waveforms_b.shape[0] > max_samples:
        indices = np.random.choice(waveforms_b.shape[0], max_samples, replace=False)
        waveforms_b = waveforms_b[indices]
    
    # Compute median templates
    template_a = np.median(waveforms_a, axis=0)
    template_b = np.median(waveforms_b, axis=0)
    
    # Pearson correlation
    corr = np.corrcoef(template_a, template_b)[0, 1]
    
    return corr


def compute_merged_isi_violation_rate(
    spike_times_a: np.ndarray,
    spike_times_b: np.ndarray,
    refractory_period_ms: float = 2.0,
) -> float:
    """
    Compute ISI violation rate if two clusters are merged.
    
    Args:
        spike_times_a: Spike times in seconds
        spike_times_b: Spike times in seconds
        refractory_period_ms: Refractory period in ms
        
    Returns:
        Violation rate (fraction of ISIs < refractory period)
    """
    # Merge and sort
    merged_times = np.sort(np.concatenate([spike_times_a, spike_times_b]))
    
    if len(merged_times) < 2:
        return 0.0
    
    # Compute ISIs
    isis = np.diff(merged_times) * 1000  # Convert to ms
    
    # Count violations
    violations = np.sum(isis < refractory_period_ms)
    violation_rate = violations / len(isis) if len(isis) > 0 else 0.0
    
    return violation_rate
