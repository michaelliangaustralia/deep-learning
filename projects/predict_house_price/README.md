# Spaceship Titanic Kaggle Competition

Simple binary classification task for a Kaggle competition.

## Submission

`submission/submission.csv`: The predictions from the model which achieves 3.88596 accuracy on the unseen competition test set.
`submission/house_price_100_epoch.pth`: The trained model weights.

## Model

The model is a naive 5-layer feed forward neural network and is trained for 100 epochs with a batch size of 16 across a 90%-10% train-test split of the training dataset. 10% dropout is applied to the middle 2 layers of the network.

## Resources
- [Spaceship Titanic Kaggle Competition](https://www.kaggle.com/competitions/spaceship-titanic/overview)