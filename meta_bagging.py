import time
import numpy as np

class MetaBagging:
    """ Meta Bagging

    It's a classic approach used for Kaggle competition, specially for challenges where variance must be as small as possible.
    Bagging is very powerful to limit this phenomenon. Instead of stacking a "simple" model as a tree, Meta Bagging uses Meta learner.
    A Meta Learner is a model which uses features from the training dataset and predictions (for this data) of a Weak learner.
    This approach try to limit the prediction errors using capacity of distinct models: Remember the No free lunch theorem !

    The choice of the training dataset for the weak and meta learner is very important. Both must be trained completely separeted.
    Data used by meta learner mustn't be used by weak learner and vice versa.

    The idea:
        1) 2 dataset: Training dataset + test dataset (no validation dataset)
        2) Bootstrap of the training dataset: By this method, ~66% of all distinct data in the training dataset are represented in the subset
        3) Out of bag: All data not represented in the bootstrap dataset (~34%)

        4) Weak learner learns on out of bag dataset (~34% of all data)
        5) Weak learner predicts value of the Bootstrap dataset AND the test dataset

        6) Meta learner learns on Bootstrap dataset where data is augmented by weak learner predictions (~66% of all data)
        7) Meta Learner predicts value for the test dataset where data is augmented by weak learner predictions

        8) Predictions are added to the previous predictions and the value is divided by the number of model in the bagging

        9) Repeat 2-8 nbr_iteration times

    Example:
        model = MetaBagging(weak_learner=["RandomForestClassifier", {"n_estimators": 100, "max_features": 0.9}, {}, "predict_proba" ],
                            meta_learner=["XGBClassifier", {"n_estimators":60, "max_depth": 11}, {"eval_metric": "mlogloss"}, "predict_proba])

        preds = model.prediction(X_train, Y_train, X_test, 200)
    """

    def __init__(self, weak_learner=None, meta_learner=None):
        """
        :param weak_learner: list
            Parameters of the weak learner:
                List[0]: string
                    Name of the model from SKLearn library (ex: "RandomForestRegressor") - Class must be imported by "from x import y"
                List[1]: dict
                    Parameters of the model (ex: {'nb_estimators': 10})
                List[2]: dict
                    Parameters of the fit() function (if parameters there are) (ex: {'eval_metric': "mlogloss"}) EXCEPT train_data et label_data
                List[3]: string
                    Name of the function used to predict values (ex: "predict" or "predict_proba")

        :param meta_learner: list
            Parameters of the meta learner:
                Same configuration as weak learner
        """

        self.weak_learner_model = weak_learner[0]
        self.weak_learner_params = weak_learner[1]
        self.weak_learner_training_function_params = weak_learner[2]
        self.weak_learner_predict_function = weak_learner[3]

        self.meta_learner_model = meta_learner[0]
        self.meta_learner_params = meta_learner[1]
        self.meta_learner_training_function_params = meta_learner[2]
        self.meta_learner_predict_function = meta_learner[3]

    def prediction(self, X_train, Y_train, X_test,  iteration=10):
        """
        :param X_train: numpy array
            Data of the training dataset

        :param Y_train: numpy array
            Label of the training dataset

        :param X_test: numpy array
            Data of the test dataset (real test not validation !)

        :param iteration: int
            Number of iteration, i.e number of model in the bagging
            The function doesn't use the 0-value iteration so the real iteration number is iteration - 1

        :return: preds: numpy array
            Predictions for the X_test dataset
        """

        preds = None

        for ite in range(1, iteration):
            print("Iteration: " + str(ite) + "/" + str(iteration))
            time_init = time.time()

            # Bootstrap + OOB
            index_bootstrap = np.random.choice(X_train.shape[0], X_train.shape[0])
            index_oob = np.setdiff1d([i for i in range(X_train.shape[0])], index_bootstrap)

            # OOB dataset
            x_train_weak = X_train[index_oob, ]
            y_train_weak = Y_train[index_oob]

            # Weak learner training
            print("Initialization weak learner:" + self.weak_learner_model)
            # globals searchs an element with the associated name and call this element
            # globals()[self.weak_learner_model] = <RandomForestRegressor> pointer for example
            weak_learner = globals()[self.weak_learner_model](**self.weak_learner_params)
            weak_learner.fit(x_train_weak, np.asarray(y_train_weak), **self.weak_learner_training_function_params)
            print("Model fitted")

            # Bootstrap dataset
            x_train_meta = X_train[index_bootstrap, ]
            y_train_meta = Y_train[index_bootstrap]

            # Weak learner predictions
            # getattr searchs a variable of an object (variable, method...) by name and call it
            # getattr(weak_learner, self.weak_learner_predict_function) = weak_learner.<predict_proba> for example
            pred_train_weak = getattr(weak_learner, self.weak_learner_predict_function)(x_train_meta)
            pred_test_weak = getattr(weak_learner, self.weak_learner_predict_function)(X_test)
            print("Weak Model prediction - Successful")

            # Meta learner training
            print("Initialization meta learner:" + self.meta_learner_model)
            meta_learner = globals()[self.meta_learner_model](**self.meta_learner_params)
            meta_learner.fit(np.concatenate((x_train_meta, pred_train_weak), axis=1), y_train_meta, **self.meta_learner_training_function_params)
            print("Model fitted")

            # Meta learner predictions
            pred_test_meta = getattr(meta_learner, self.meta_learner_predict_function)(np.concatenate((X_test, pred_test_weak), axis=1))
            print("Meta Model prediction - Successful")

            preds = preds + pred_test_meta
            print("Time for the iteration: " + str(time.time() - time_init))

        preds /= (iteration - 1)

        return preds
