"""driverbench: driver-vs-marker benchmark and the timing-constrained interventional counterfactual method.

Main entry points
  driverbench.evaluate.contexts / metrics       decision contexts and the driver-vs-marker metrics
  driverbench.method_timing.score_timing_cf_v2  our method (frozen version 2)
  driverbench.baselines.SCORERS                 all baselines
  driverbench.competitors                       CellRank 2 / CellOracle / CEFCON adapters
  driverbench.simulate, driverbench.networks    BoolODE networks and simulation
  driverbench.realdata                          shared preprocessing + literature evaluation for real data
"""
