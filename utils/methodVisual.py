import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(7, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Helper function
def box(x, y, w, h, text, color):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        linewidth=2,
        edgecolor='black',
        facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=12)

def arrow(x1, y1, x2, y2):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2)
    )

# Colors
c_input = "#A8D5E2"
c_rank = "#BDE5B8"
c_fair = "#F7C6C7"
c_high = "#F28C8C"
c_out = "#DDDDDD"

# INPUT
box(2.5, 10.8, 5, 0.9, "User–Item Interactions", c_input)

# RANKING
box(2.5, 9.5, 5, 0.8, "BPR Training", c_rank)
box(2.5, 8.4, 5, 0.8, "Generate Top-K Rankings", c_rank)

# FAIRNESS TITLE
ax.text(2.7, 7.6, "Global Exposure-Aware Module (GEA-MF)",
        fontsize=12, weight='bold')

# FAIRNESS BLOCKS
box(2.5, 7.2, 5, 0.8, "Compute Exposure (E₀, E₁)", c_fair)
box(2.5, 6.1, 5, 0.8, "Compute Imbalance Δ", c_fair)
box(2.5, 5.0, 5, 0.8, "Global Correction", c_high)

# OUTPUT
box(2.5, 3.7, 5, 0.8, "Updated Item Embeddings", c_out)

# ARROWS
arrow(5, 10.8, 5, 10.3)
arrow(5, 9.5, 5, 9.2)
arrow(5, 8.4, 5, 8.0)
arrow(5, 7.2, 5, 6.8)
arrow(5, 6.1, 5, 5.7)
arrow(5, 5.0, 5, 4.6)

# LOOP ARROW (MUCH CLEANER)
ax.annotate("",
    xy=(2.5, 9.5), xytext=(2.5, 3.7),
    arrowprops=dict(arrowstyle="->", linestyle='dashed')
)
ax.text(1.0, 6.5, "Next epoch", fontsize=10)

# Save
plt.savefig("gea_mf_clean.png", dpi=300, bbox_inches='tight')
plt.show()