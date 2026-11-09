# Data attribution

The files in `data/marts/` are derived from the **Microsoft Azure Public Dataset V2**
(https://github.com/Azure/AzurePublicDataset), © Microsoft Corporation, licensed under
**Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Changes made: VM lifetimes were split into hourly usage, priced at Azure list prices
(see `src/cca/pricing.py`), and aggregated into reporting tables. Readable aliases
(SUB-0001 ...) were added next to Microsoft's already-encrypted subscription identifiers. Source paper: Cortez et al., "Resource
Central: Understanding and Predicting Workloads for Improved Resource Management in Large
Cloud Platforms", SOSP 2017. The code in this repository is MIT-licensed (see LICENSE).
