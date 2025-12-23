# M-STAR
M-STAR is a Multi-Scale Spatio-temporal Attention Refinement Network for short-term citywide taxi inflow–outflow prediction on grid-discretized urban networks.

![M-STAR employs a unified encoder--decoder pipeline that transforms raw grid-level origin--destination data into multi-step traffic forecasts. The architecture consists of four sequential components: (1) multi-source data embedding, which projects numerical, temporal, and spatial information into a shared latent space; (2) a hierarchical spatiotemporal encoder that refines representations through stacked blocks; (3) cross-layer skip connections for feature fusion; and (4) a prediction decoder that outputs future inflow--outflow trajectories. ](figure/merged_grid_maps.png)
