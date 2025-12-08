### Metric

**Accuracy** measures the proportion of correctly classified examples and is defined as:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$




### Results

| Model | Val Accuracy| Test Accuracy|
|-------------|-----------|-----------|
| **GCN - from paper**              | -   | **70.3**   |
| GCN - from paper our replication  | 0.718   | 0.691   |
| GCN - without edges           | 0.572   | 0.571   |
| Graph Isomorphism Network     | 0.666   | 0.648   |
| Graph Attention Network       | 0.748   | 0.728   |
| Graph Sample and aggregate    | 0.710   | 0.707   |
| GraphTransformer              | 0.724   | 0.694   |
| MLP without edges             | 0.444   | 0.402   |

### Conclusions

Across all experiments on the Citeseer dataset, models that incorporate graph structure consistently outperform those that ignore it, confirming the importance of relational information in semi-supervised node classification. Our replication of the original GCN achieved slightly lower accuracy than reported in the paper but still demonstrated strong performance relative to baselines. Among all models evaluated, the Graph Attention Network achieved the highest accuracy, suggesting that adaptive, attention-based neighbor weighting is more effective than fixed aggregation. GraphSAGE and GraphTransformer also performed competitively, while GIN lagged somewhat, reflecting its sensitivity to hyperparameters in smaller citation networks. In contrast, models without edges—GCN without edges and an MLP—performed significantly worse, emphasizing that leveraging graph connectivity is crucial for achieving high performance in this task.