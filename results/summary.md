# Retrieval evaluation summary

200 recipes × 5 query sets (see reports/report.pdf). Rerank/agent rows: 50 queries per set (image+text rerank: 25).

### Recall@1

| Pipeline (eval run) | Method | Title | Paraphrase | Spanish | Bengali | Chinese | Avg |
|---|---|---|---|---|---|---|---|
| original_stack | baseline: dense | 0.935 | 0.650 | 0.245 | 0.140 | 0.100 | 0.414 |
| original_stack | baseline: dense + rerank (image+text) | 0.880 | 0.720 | 0.160 | 0.160 | 0.080 | 0.400 |
| original_stack | bm25 only | 0.820 | 0.690 | 0.015 | 0.005 | 0.005 | 0.307 |
| original_stack | improved: hybrid (dense + bm25, RRF) | 0.895 | 0.735 | 0.110 | 0.140 | 0.105 | 0.397 |
| original_stack | improved: hybrid (bm25 weight 0.5) | 0.905 | 0.735 | 0.115 | 0.140 | 0.105 | 0.400 |
| original_stack | improved: hybrid + rerank (text) | 0.480 | 0.320 | 0.100 | 0.120 | 0.080 | 0.220 |
| improved_reembedded | baseline: dense | 0.965 | 0.750 | 0.260 | 0.165 | 0.130 | 0.454 |
| improved_reembedded | bm25 only | 0.820 | 0.690 | 0.015 | 0.005 | 0.005 | 0.307 |
| improved_reembedded | improved: hybrid (dense + bm25, RRF) | 0.920 | 0.765 | 0.100 | 0.165 | 0.140 | 0.418 |
| improved_reembedded | improved: gated hybrid (bm25 if coverage >= 0.8) | 0.920 | 0.765 | 0.255 | 0.165 | 0.130 | 0.447 |
| improved_reembedded | improved: hybrid + rerank (text) | 0.980 | 0.860 | 0.460 | 0.320 | 0.280 | 0.580 |
| improved_reembedded | improved: gated hybrid + rerank (text) | 0.980 | 0.860 | 0.480 | 0.320 | 0.280 | 0.584 |
| qwen_vl | qwen3-vl: dense | 0.960 | 0.770 | 0.415 | 0.335 | 0.470 | 0.590 |
| qwen_vl | qwen3-vl: hybrid (dense + bm25, RRF) | 0.935 | 0.795 | 0.150 | 0.335 | 0.465 | 0.536 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (text) | 0.980 | 0.860 | 0.600 | 0.500 | 0.660 | 0.720 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (image+text) | 1.000 | 0.920 | 0.560 | 0.360 | 0.480 | 0.664 |
| agentic | agentic: tool-call rewrite + hybrid | 0.780 | 0.540 | 0.280 | 0.400 | 0.320 | 0.464 |
| agentic | agentic: tool-call rewrite + hybrid + rerank (text) | 0.940 | 0.780 | 0.540 | 0.440 | 0.480 | 0.636 |
| agentic | agentic (thinking): rewrite + hybrid + rerank (text) | 0.920 | 0.760 | 0.520 | 0.440 | 0.480 | 0.624 |
| final | final: qwen3-vl gated hybrid | 0.935 | 0.795 | 0.390 | 0.335 | 0.470 | 0.585 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (text) | 0.980 | 0.860 | 0.720 | 0.500 | 0.680 | 0.748 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text) | 1.000 | 0.920 | 0.520 | 0.360 | 0.560 | 0.672 |
| final | final: agent + qwen3-vl gated hybrid + qwen3-vl rerank (text) | 0.940 | 0.800 | 0.600 | 0.480 | 0.520 | 0.668 |

### MRR@10
### Recall@10

| Pipeline (eval run) | Method | Title | Paraphrase | Spanish | Bengali | Chinese | Avg |
|---|---|---|---|---|---|---|---|
| original_stack | baseline: dense | 1.000 | 0.945 | 0.590 | 0.355 | 0.370 | 0.652 |
| original_stack | baseline: dense + rerank (image+text) | 1.000 | 0.960 | 0.560 | 0.240 | 0.360 | 0.624 |
| original_stack | bm25 only | 0.980 | 0.940 | 0.075 | 0.005 | 0.015 | 0.403 |
| original_stack | improved: hybrid (dense + bm25, RRF) | 0.995 | 0.955 | 0.410 | 0.355 | 0.380 | 0.619 |
| original_stack | improved: hybrid (bm25 weight 0.5) | 0.995 | 0.960 | 0.520 | 0.355 | 0.380 | 0.642 |
| original_stack | improved: hybrid + rerank (text) | 0.940 | 0.700 | 0.380 | 0.300 | 0.220 | 0.508 |
| improved_reembedded | baseline: dense | 1.000 | 0.960 | 0.600 | 0.360 | 0.380 | 0.660 |
| improved_reembedded | bm25 only | 0.980 | 0.940 | 0.075 | 0.005 | 0.015 | 0.403 |
| improved_reembedded | improved: hybrid (dense + bm25, RRF) | 0.995 | 0.965 | 0.485 | 0.360 | 0.380 | 0.637 |
| improved_reembedded | improved: gated hybrid (bm25 if coverage >= 0.8) | 0.995 | 0.965 | 0.565 | 0.360 | 0.380 | 0.653 |
| improved_reembedded | improved: hybrid + rerank (text) | 1.000 | 0.960 | 0.540 | 0.360 | 0.360 | 0.644 |
| improved_reembedded | improved: gated hybrid + rerank (text) | 1.000 | 0.960 | 0.580 | 0.360 | 0.360 | 0.652 |
| qwen_vl | qwen3-vl: dense | 1.000 | 0.980 | 0.775 | 0.670 | 0.835 | 0.852 |
| qwen_vl | qwen3-vl: hybrid (dense + bm25, RRF) | 0.995 | 0.980 | 0.625 | 0.670 | 0.840 | 0.822 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (text) | 1.000 | 0.960 | 0.740 | 0.700 | 0.900 | 0.860 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (image+text) | 1.000 | 0.960 | 0.720 | 0.760 | 0.880 | 0.864 |
| agentic | agentic: tool-call rewrite + hybrid | 1.000 | 0.880 | 0.720 | 0.660 | 0.740 | 0.800 |
| agentic | agentic: tool-call rewrite + hybrid + rerank (text) | 1.000 | 0.940 | 0.780 | 0.700 | 0.820 | 0.848 |
| agentic | agentic (thinking): rewrite + hybrid + rerank (text) | 1.000 | 0.960 | 0.720 | 0.760 | 0.760 | 0.840 |
| final | final: qwen3-vl gated hybrid | 0.995 | 0.980 | 0.735 | 0.670 | 0.835 | 0.843 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (text) | 1.000 | 0.960 | 0.840 | 0.700 | 0.920 | 0.884 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text) | 1.000 | 0.960 | 0.840 | 0.760 | 0.920 | 0.896 |
| final | final: agent + qwen3-vl gated hybrid + qwen3-vl rerank (text) | 1.000 | 0.980 | 0.880 | 0.780 | 0.860 | 0.900 |

### Mean latency per query (ms)

| Pipeline (eval run) | Method | Title | Paraphrase | Spanish | Bengali | Chinese | Avg |
|---|---|---|---|---|---|---|---|
| original_stack | baseline: dense | 19 | 19 | 19 | 20 | 20 | 20 |
| original_stack | baseline: dense + rerank (image+text) | 16892 | 16019 | 15640 | 12213 | 13963 | 14945 |
| original_stack | bm25 only | 1 | 0 | 1 | 0 | 0 | 0 |
| original_stack | improved: hybrid (dense + bm25, RRF) | 20 | 20 | 21 | 20 | 20 | 20 |
| original_stack | improved: hybrid (bm25 weight 0.5) | 20 | 20 | 20 | 20 | 20 | 20 |
| original_stack | improved: hybrid + rerank (text) | 562 | 570 | 611 | 689 | 589 | 604 |
| improved_reembedded | baseline: dense | 17 | 18 | 18 | 18 | 17 | 18 |
| improved_reembedded | bm25 only | 1 | 0 | 1 | 0 | 0 | 0 |
| improved_reembedded | improved: hybrid (dense + bm25, RRF) | 18 | 18 | 19 | 18 | 18 | 18 |
| improved_reembedded | improved: gated hybrid (bm25 if coverage >= 0.8) | 18 | 18 | 18 | 17 | 18 | 18 |
| improved_reembedded | improved: hybrid + rerank (text) | 553 | 546 | 603 | 683 | 579 | 593 |
| improved_reembedded | improved: gated hybrid + rerank (text) | 544 | 552 | 587 | 683 | 581 | 589 |
| qwen_vl | qwen3-vl: dense | 47 | 48 | 47 | 47 | 47 | 47 |
| qwen_vl | qwen3-vl: hybrid (dense + bm25, RRF) | 48 | 48 | 48 | 48 | 48 | 48 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (text) | 935 | 935 | 969 | 1060 | 959 | 972 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (image+text) | 2708 | 2722 | 2709 | 2825 | 2703 | 2733 |
| agentic | agentic: tool-call rewrite + hybrid | 4365 | 4893 | 4464 | 4538 | 5273 | 4707 |
| agentic | agentic: tool-call rewrite + hybrid + rerank (text) | 4906 | 5409 | 5027 | 5117 | 5854 | 5263 |
| agentic | agentic (thinking): rewrite + hybrid + rerank (text) | 14152 | 14773 | 16136 | 17365 | 19527 | 16391 |
| final | final: qwen3-vl gated hybrid | 48 | 49 | 48 | 49 | 49 | 49 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (text) | 932 | 935 | 975 | 1057 | 958 | 971 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text) | 2708 | 2732 | 2766 | 2873 | 2738 | 2763 |
| final | final: agent + qwen3-vl gated hybrid + qwen3-vl rerank (text) | 5307 | 5796 | 5410 | 5477 | 6197 | 5637 |

### Queries evaluated (n)

| Pipeline (eval run) | Method | Title | Paraphrase | Spanish | Bengali | Chinese | Avg |
|---|---|---|---|---|---|---|---|
| original_stack | baseline: dense | 200 | 200 | 200 | 200 | 200 | 200 |
| original_stack | baseline: dense + rerank (image+text) | 25 | 25 | 25 | 25 | 25 | 25 |
| original_stack | bm25 only | 200 | 200 | 200 | 200 | 200 | 200 |
| original_stack | improved: hybrid (dense + bm25, RRF) | 200 | 200 | 200 | 200 | 200 | 200 |
| original_stack | improved: hybrid (bm25 weight 0.5) | 200 | 200 | 200 | 200 | 200 | 200 |
| original_stack | improved: hybrid + rerank (text) | 50 | 50 | 50 | 50 | 50 | 50 |
| improved_reembedded | baseline: dense | 200 | 200 | 200 | 200 | 200 | 200 |
| improved_reembedded | bm25 only | 200 | 200 | 200 | 200 | 200 | 200 |
| improved_reembedded | improved: hybrid (dense + bm25, RRF) | 200 | 200 | 200 | 200 | 200 | 200 |
| improved_reembedded | improved: gated hybrid (bm25 if coverage >= 0.8) | 200 | 200 | 200 | 200 | 200 | 200 |
| improved_reembedded | improved: hybrid + rerank (text) | 50 | 50 | 50 | 50 | 50 | 50 |
| improved_reembedded | improved: gated hybrid + rerank (text) | 50 | 50 | 50 | 50 | 50 | 50 |
| qwen_vl | qwen3-vl: dense | 200 | 200 | 200 | 200 | 200 | 200 |
| qwen_vl | qwen3-vl: hybrid (dense + bm25, RRF) | 200 | 200 | 200 | 200 | 200 | 200 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (text) | 50 | 50 | 50 | 50 | 50 | 50 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (image+text) | 25 | 25 | 25 | 25 | 25 | 25 |
| agentic | agentic: tool-call rewrite + hybrid | 50 | 50 | 50 | 50 | 50 | 50 |
| agentic | agentic: tool-call rewrite + hybrid + rerank (text) | 50 | 50 | 50 | 50 | 50 | 50 |
| agentic | agentic (thinking): rewrite + hybrid + rerank (text) | 25 | 25 | 25 | 25 | 25 | 25 |
| final | final: qwen3-vl gated hybrid | 200 | 200 | 200 | 200 | 200 | 200 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (text) | 50 | 50 | 50 | 50 | 50 | 50 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text) | 25 | 25 | 25 | 25 | 25 | 25 |
### Mean latency per query (ms)

| Pipeline (eval run) | Method | Title | Paraphrase | Spanish | Bengali | Chinese | Avg |
|---|---|---|---|---|---|---|---|
| original_stack | baseline: dense | 19 | 19 | 19 | 20 | 20 | 20 |
| original_stack | baseline: dense + rerank (image+text) | 16892 | 16019 | 15640 | 12213 | 13963 | 14945 |
| original_stack | bm25 only | 1 | 0 | 1 | 0 | 0 | 0 |
| original_stack | improved: hybrid (dense + bm25, RRF) | 20 | 20 | 21 | 20 | 20 | 20 |
| original_stack | improved: hybrid (bm25 weight 0.5) | 20 | 20 | 20 | 20 | 20 | 20 |
| original_stack | improved: hybrid + rerank (text) | 562 | 570 | 611 | 689 | 589 | 604 |
| improved_reembedded | baseline: dense | 17 | 18 | 18 | 18 | 17 | 18 |
| improved_reembedded | bm25 only | 1 | 0 | 1 | 0 | 0 | 0 |
| improved_reembedded | improved: hybrid (dense + bm25, RRF) | 18 | 18 | 19 | 18 | 18 | 18 |
| improved_reembedded | improved: gated hybrid (bm25 if coverage >= 0.8) | 18 | 18 | 18 | 17 | 18 | 18 |
| improved_reembedded | improved: hybrid + rerank (text) | 553 | 546 | 603 | 683 | 579 | 593 |
| improved_reembedded | improved: gated hybrid + rerank (text) | 544 | 552 | 587 | 683 | 581 | 589 |
| qwen_vl | qwen3-vl: dense | 47 | 48 | 47 | 47 | 47 | 47 |
| qwen_vl | qwen3-vl: hybrid (dense + bm25, RRF) | 48 | 48 | 48 | 48 | 48 | 48 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (text) | 935 | 935 | 969 | 1060 | 959 | 972 |
| qwen_vl | qwen3-vl: hybrid + qwen3-vl rerank (image+text) | 2708 | 2722 | 2709 | 2825 | 2703 | 2733 |
| agentic | agentic: tool-call rewrite + hybrid | 4365 | 4893 | 4464 | 4538 | 5273 | 4707 |
| agentic | agentic: tool-call rewrite + hybrid + rerank (text) | 4906 | 5409 | 5027 | 5117 | 5854 | 5263 |
| agentic | agentic (thinking): rewrite + hybrid + rerank (text) | 14152 | 14773 | 16136 | 17365 | 19527 | 16391 |
| final | final: qwen3-vl gated hybrid | 48 | 49 | 48 | 49 | 49 | 49 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (text) | 932 | 935 | 975 | 1057 | 958 | 971 |
| final | final: qwen3-vl gated hybrid + qwen3-vl rerank (image+text) | 2708 | 2732 | 2766 | 2873 | 2738 | 2763 |
| final | final: agent + qwen3-vl gated hybrid + qwen3-vl rerank (text) | 5307 | 5796 | 5410 | 5477 | 6197 | 5637 |

