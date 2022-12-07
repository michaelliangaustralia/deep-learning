# Third party imports
import datasets
import IPython

# Native imports.
import sys
sys.path.append('../..') # Configuring sys path to enable imports from parent folder.
import common.utils as common_utils

ds = datasets.load_dataset('csv', data_files="train.csv")['train']

# Change datatype of columns
categorical_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
ds = common_utils.cast_hf_dataset_columns(ds, datasets.Value("string"), categorical_columns)

# One hot encode all categorical columns.
dict_maps = common_utils.get_multiple_dict_maps(ds, categorical_columns)

one_hot_encode_dict_map_list = []
ds = ds.map(common_utils.one_hot_encode, fn_kwargs= {
    'dict_maps': dict_maps,
    'column_names': categorical_columns
    }, num_proc=4)

# Remove all categorical and noisy columns.
ds = ds.remove_columns(categorical_columns)
noisy_columns = ['Id']
ds = ds.remove_columns(noisy_columns)

# Fill integer nulls with 0.
integer_columns = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',  'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

ds = ds.map(common_utils.replace_column_value, fn_kwargs= {
    'column_names': integer_columns,
    'fill_value': 0,
    'replace_value': None,
    }, num_proc=1)

# Save to csv
ds.save_to_disk('train_processed')