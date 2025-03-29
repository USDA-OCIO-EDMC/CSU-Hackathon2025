import optuna
from torch.utils.data import DataLoader
from utils.data_preparation import prepare_data
from utils.training_module import get_model, training_loop, evaluation_loop

# Precompute the data once for all trials.
global_train_ds, global_val_ds, _ = prepare_data(oversample=True)

def objective(trial):
    # Sample hyperparameters.
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 1, 2)
    batch_size = trial.suggest_int("batch_size", 64, 78)
    
    train_loader = DataLoader(global_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(global_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model, loss_fn, optimizer = get_model(lr=lr)
    training_loop(model, train_loader, loss_fn, optimizer, num_epochs=epochs)
    val_loss = evaluation_loop(model, val_loader, loss_fn)
    return val_loss
