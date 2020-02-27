# TimeSeriesTrain
# small d3m test dataset
# need to evaluate outputs of methods or objects they should affect

# datetime interpolation
# - no duplicates
# -correctly reindexed (not too many NAs - maybe assertion)
# -correct cols interpolated
# correct ordering of cols

# pad ts
# correct columns interpolated / standardized / changed

# add prev target
# for train/val/test case make sure alignment is correct
# handles columns w/ NAs appropriately

# sample missing tgts
# correct indices are replaced
# correct scaling / standardization

# properties
# idxs align w/ constructor arguments
# missing tgt vals make sense w/ original NAs + reindexing (maybe assertion??)


# TimeSeriesTest

# preproces new test data
# updates other indices correctly
# applicaton of scaler makes sense w/ covariate col idxs (could be assertion)


# Utils

# robust reindex
# w/ different tolerance values, reindexing never should produce completely empty dataset