# Local Dependencies
# ------------------
from features import RandomForestFeaturesNegation


def test_rf_feature_names(n2c2_small_collection):
    rf = RandomForestFeaturesNegation("n2c2")
    features = rf.fit_transform(n2c2_small_collection)
    
    assert len(rf.get_feature_names()) == features.shape[1]