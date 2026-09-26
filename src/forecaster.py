import torch
from pathlib import Path
from aurora import Batch, rollout

class AuroraForecaster:
    """
    A dedicated class to handle model forward passes, 
    autoregressive rollouts, and saving outputs.
    """
    def __init__(self, model, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        print(f"Initializing Aurora Forecaster on {self.device}...")
        
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def predict_step(self, input_batch: Batch) -> Batch:
        """
        Executes a single forward pass.
        """
        input_batch = input_batch.to(self.device)

        with torch.inference_mode():
            prediction = self.model.forward(input_batch)
            
        return prediction

    def predict_rollout(
        self, 
        initial_batch: Batch, 
        steps: int, 
        fine_lead_times: list[float] = None,
        save_path: Path = None
    ) -> list[Batch]:
        """
        Executes an autoregressive rollout to predict far into the future.
        
        Args:
            initial_batch: The starting [t-6, t0] batch.
            steps: Number of main 6-hour autoregressive jumps.
            fine_lead_times: Sub-hourly outputs per jump. Defaults to hourly [1.0 ... 6.0].
            save_path: Optional path to immediately save the resulting predictions.
        """
        kwargs = {"steps": steps}
        if fine_lead_times is not None:
            kwargs["fine_lead_times"] = fine_lead_times
        elif getattr(self.model, "variable_lead_time", False):
            # Default to Aurora 1.5's behavior if not provided and supported
            kwargs["fine_lead_times"] = [6.0]

        current_batch = initial_batch.to(self.device)
        num_sub_steps = len(kwargs.get("fine_lead_times", [6.0]))
        total_preds = steps * num_sub_steps
        
        torch.cuda.empty_cache()

        print(f"Starting {steps}-step AR rollout with {num_sub_steps} sub-steps per AR step.")
        print(f"Total generated predictions will be: {total_preds}")

        with torch.inference_mode():
            predictions = [
                pred.to("cpu") 
                for pred in rollout(
                    self.model, 
                    current_batch, 
                    **kwargs
                )
            ]
                
        if save_path:
            self.save_predictions(predictions, save_path)
            
        return predictions

    def save_predictions(self, predictions: Batch | list[Batch], save_path: Path):
        """
        Saves the predicted batches to the hard drive.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(predictions, save_path)
        print(f"Predictions successfully saved to {save_path}")