# Bayesian calibration

The method is explained in the
[user guide](../guide/bayes-gscale.md). It is run by
`Observation.gscale(bayes=True, ...)`, whose parameters are in the
[Observation reference](observation.md); with `quiet=False` it returns
the result below.

::: difmapy.bayescal.BayesGainResult
    options:
      members:
        - factors
        - best_model
        - summary
        - findings
        - station_table
        - to_dict
        - plot
        - write
        - save_caltable
