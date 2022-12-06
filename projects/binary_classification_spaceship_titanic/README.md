# Predict House Price Kaggle Competition

Kaggle competition.

## Submission

`submission/submission.csv`: The predictions from the model which achieves 0.77881 accuracy on the unseen competition test set.
`submission/spaceship_titanic_999_epoch.pth`: The trained model weights.

## Model

The model is a naive 3-layer feed forward neural network and is trained for 1000 epochs with a batch size of 32 across a 90%-10% train-test split of the training dataset.

## Resources
- [House Price Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)