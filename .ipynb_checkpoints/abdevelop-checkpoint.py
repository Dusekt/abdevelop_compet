import torch
import numpy as np
import joblib
from sklearn.linear_model import Ridge, ElasticNet
from collections import defaultdict

from src.models import AntibodyModel2, AntibodyModel3, AntibodyModel4, AntibodyModel5

class EnsembleModel:
    def __init__(self, task_id: int, tasks_path="data/ens_model_weights/all_meta_models.pkl",
                 weights_root="data/model_weights", device="cpu"):
        self.task_id = task_id
        self.device = device

        # Load trained Ridge/ElasticNet parameters
        tasks = joblib.load(tasks_path)
        task_info = tasks[task_id]
        params = task_info["best_params"]
        coef_ = task_info["coef_"]

        # Build 20 base models in correct order (5 folds × models 2–5)
        self.base_models = []
        for fold in range(5):
            for m in range(2, 6):
                weights_path = f"{weights_root}/cv_fold_{fold}_model{m}_{task_id}.pt"
                if m == 2:
                    model = AntibodyModel2()
                elif m == 3:
                    model = AntibodyModel3()
                elif m == 4:
                    model = AntibodyModel4()
                elif m == 5:
                    model = AntibodyModel5()
                else:
                    raise ValueError(f"Unknown model index: {m}")

                # Load weights
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict, strict=True)
                model.to(device)
                model.eval()
                self.base_models.append(model)

        # Build ridge (or elasticnet) ensemble head
        if params["model_type"] == "ridge":
            self.ensemble_head = Ridge(
                alpha=params["alpha"],
                fit_intercept=params["fit_intercept"],
                solver=params["solver"]
            )
        elif params["model_type"] == "elasticnet":
            self.ensemble_head = ElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                fit_intercept=params["fit_intercept"]
            )
        else:
            raise ValueError(f"Unknown model_type: {params['model_type']}")

        # Assign trained coefficients
        self.ensemble_head.coef_ = coef_
        self.ensemble_head.intercept_ = task_info.get("intercept_", 0.0)

    @torch.no_grad()
    def __call__(self, embeddings, descriptors):
        """
        embeddings: np.ndarray or torch.Tensor of shape (N, E)
        descriptors: np.ndarray or torch.Tensor of shape (N, D)
        returns: np.ndarray of shape (N,)
        """
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(descriptors):
            descriptors = torch.tensor(descriptors, dtype=torch.float32, device=self.device)

        preds = []
        for model in self.base_models:
            out = model(embeddings, descriptors)
            preds.append(out.cpu().numpy().reshape(-1))  # flatten each model's output

        preds = np.stack(preds, axis=1)  # shape (N, 20)

        # Feed to trained ridge/elasticnet
        final_pred = self.ensemble_head.predict(preds)
        return final_pred
if __name__ == '__main__':
    # Instantiate for a specific task
    model = EnsembleModel(task_id=2)
    # Example input
    N, E, D = 8, 480, 30
    embeddings = np.random.randn(N, E)
    descriptors = np.random.randn(N, D)
    
    # Predict
    y_pred = model(embeddings, descriptors)
    print("Predictions:", y_pred.shape, y_pred[:5])
