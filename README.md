# Animal Classifier MLOps Pipeline

This project implements an end-to-end MLOps workflow for training and evaluating a PyTorch-based animal image classifier with ZenML and MLflow. The pipeline covers data preparation, model training, and evaluation while logging artifacts and metrics to MLflow for experiment tracking.

## Prerequisites

- macOS or Linux with Python 3.12+
- `curl` (for installation convenience)
- Git

> **Note**  
> Commands below assume the repository root as the working directory.

## Environment Setup

### 1. Install the `uv` Package Manager

`uv` provides fast dependency management and isolated environments.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ensure the installer adds `uv` to your `PATH` (reopen your shell if needed) and verify the install:

```bash
uv --version
```

### 2. Create the Virtual Environment and Install Dependencies

```bash
uv sync
```

`uv sync` creates `.venv` (if it does not already exist) and installs every dependency listed in `pyproject.toml` / `uv.lock` (PyTorch, torchvision, ZenML, MLflow, etc.).  
Run future commands with `uv run …` or activate the environment manually via `source .venv/bin/activate`.

## ZenML & MLflow Setup

ZenML orchestrates the pipeline and needs an experiment tracker. A minimal first-time setup is:

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local             # start the local server
mkdir -p mlruns
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri='file:./mlruns'
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml stack register local_mlflow_stack -o default -a default -D default -e mlflow_tracker --set
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml stack describe
```

If `zenml login --local` reports `TCP port 8237 is not available`, free the port (`lsof -i :8237`) or tell ZenML to use any free port:  
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local --port 0`

### Full ZenML Workflow (from a clean state)

1. **Check server status**
   ```bash
   OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml status
   ```
   - If the output indicates that the ZenML server is running, stop it for a clean restart:
     ```bash
     OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml logout --local
     ```

2. **Start / log into the local server**
   ```bash
   OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local
   ```
   - Port busy? release it (`lsof -i :8237`) or let ZenML pick one automatically:
     ```bash
     OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local --port 0
     ```

3. **Verify the MLflow experiment tracker**
   ```bash
   OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml experiment-tracker list
   ```
   - If `mlflow_tracker` does not appear, register it:
     ```bash
     mkdir -p mlruns
     OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri='file:./mlruns'
     ```

4. **Configure the active stack**
   ```bash
   OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml stack list
   ```
   - If `local_mlflow_stack` exists, activate it:
     ```bash
     OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml stack set local_mlflow_stack
     ```
   - Otherwise, create it with the default orchestrator/deployer/artifact store and the MLflow tracker:
     ```bash
     OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml stack register local_mlflow_stack -o default -a default -D default -e mlflow_tracker --set
     ```

5. **Confirm the stack configuration**
   ```bash
   OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml stack describe
   ```
   Ensure the output lists `mlflow_tracker` under “EXPERIMENT_TRACKER”.

6. **Initialize ZenML**
    ```bash
    zenml init
    ```
    
7. **Run the pipeline**
   ```bash
   uv run run_training.py
   ```

### Resetting ZenML State (Optional)

If you ever want to start from a completely clean slate:

1. Disconnect from and shut down the local ZenML server:
   ```bash
   OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml logout --local
   ```
2. Delete the global ZenML store (macOS path shown; use `~/.config/zenml` on Linux):
   ```bash
   rm -rf ~/Library/Application\ Support/zenml
   ```
3. Remove project-level caches if they exist:
   ```bash
   rm -rf .zen mlruns
   ```
4. Verify the reset:
   ```bash
   zenml status
   ```
   The output should report that no server is running and no active stack is configured.  
   Afterwards repeat the “Full ZenML Workflow” steps to reinitialize everything.

## Running the Training Pipeline

Execute the ZenML pipeline to prepare data, train the model, and evaluate it:

```bash
uv run run_training.py
```

Successful runs log metrics and artifacts to MLflow. To browse them, launch the MLflow UI provided by ZenML (URL shown in the CLI output for the ZenML dashboard with MLflow accessible through it).

You can manually launch the MLflow experiment tracking through this command (in a separate terminal):
```bash
uv run mlflow ui --backend-store-uri file:./mlruns --port 5000
```

## Repository Structure

- `config.yaml` – Configurable data and training parameters.
- `src/` – Pipeline steps, data handling utilities, models, and custom materializers.
- `mlruns/` – Local MLflow artifact and experiment logs (created automatically).
- `models/` – Trained model checkpoints.
- `run_training.py` – Script entry point to fire the ZenML pipeline.

## Troubleshooting Tips

- **ZenML stack errors**: Ensure the experiment tracker is registered and attached to the active stack (`zenml stack describe`).
- **GPU usage**: The pipeline defaults to CUDA. If CUDA is unavailable it falls back to CPU (warning in logs is expected).
- **Network-bound dataset preparation**: Data preparation may download images; ensure network access is available on first run.

With the environment provisioned and ZenML configured, you can iterate on the pipeline, inspect MLflow runs, and extend the project for additional MLOps stages such as model deployment or monitoring.
