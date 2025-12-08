import matplotlib.pyplot as plt

models = [
    "GCN no edges",
    "GCN (paper)",
    "GCN (our)",
    "GIN",
    "GAT",
    "GraphSAGE",
    "GraphTransformer",
]

colors = ['#ff6361', '#ffa600', '#58508d', '#58508d', '#58508d', '#58508d', '#58508d']

test_acc = [0.571, 0.703, 0.691, 0.648, 0.728, 0.707, 0.694]

plt.figure(figsize=(10,6))
plt.bar(models, test_acc, color = colors)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.ylabel("Test Accuracy", fontsize=14)
plt.title("Model Test Accuracy Comparison", fontsize=18)
plt.tight_layout()
plt.show()
