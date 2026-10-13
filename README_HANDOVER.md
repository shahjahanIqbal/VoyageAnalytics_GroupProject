# Voyage Analytics — Docker and Kubernetes handover

## Scope

These files package and run the existing Streamlit application (`app.py`) and provide a starter Kubernetes deployment. They do not create a new ML model or a FastAPI service.

## Expected location

Keep these files in the same repository root as `app.py` and `requirements.txt`:

- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `k8s/deployment.yaml`
- `k8s/service.yaml`

The repository must also include every file the app loads at runtime: saved `.joblib` model/encoders, required CSV datasets, and generated recommendation data if the Destination Planner expects it.

## 1. Build and run with Docker

Run from the repository root (the folder containing `Dockerfile`, `app.py`, and `requirements.txt`):

```powershell
docker --version
docker info
docker build -t voyage-analytics:local .
docker run --rm --name voyage-analytics-test -p 8501:8501 voyage-analytics:local
```

Open http://localhost:8501. Test both Flight Price Predictor and Destination Planner. Stop with Ctrl+C.

`docker build` packages the app; `docker run` starts a container and maps host port 8501 to container port 8501.

## 2. Use Docker Compose

```powershell
docker compose up --build
```

Open http://localhost:8501. Stop with Ctrl+C, then run:

```powershell
docker compose down
```

## 3. Kubernetes on Docker Desktop

Prerequisites:
- Docker Desktop engine is running.
- Kubernetes is enabled and running in Docker Desktop.
- `kubectl` is installed and pointed at the intended cluster.
- The image is built locally with the exact tag `voyage-analytics:local`.

Check:

```powershell
kubectl config current-context
kubectl get nodes
```

Build the image:

```powershell
docker build -t voyage-analytics:local .
```

Apply the Kubernetes files:

```powershell
kubectl apply -f k8s/
kubectl get deployments
kubectl get pods
kubectl get services
```

For a local Docker Desktop Kubernetes cluster, try:

http://localhost:30081

Inspect logs:

```powershell
kubectl logs deployment/voyage-analytics
```

Remove the deployment and Service:

```powershell
kubectl delete -f k8s/
```

For a remote/cloud cluster, push the image to a container registry and change the image reference. A local image is not automatically available to every Kubernetes cluster.

## What each file does

- `Dockerfile`: chooses Python, installs dependencies, copies the project, and starts Streamlit.
- `.dockerignore`: keeps local virtual environments, Git history, caches, and local `.env` files out of the build context.
- `docker-compose.yml`: makes local build/run repeatable with one command.
- `k8s/deployment.yaml`: asks Kubernetes to keep one Pod running, with resource limits and HTTP probes.
- `k8s/service.yaml`: exposes the app on port 30081 for local testing.

## Important checks before calling this production-ready

1. **Model compatibility:** pin versions in `requirements.txt`, especially scikit-learn, NumPy, and joblib, to versions compatible with the saved model. Compatibility has not been verified by these configuration files.
2. **Recommendation data:** confirm all runtime files exist, including generated files such as `travel_recommender_outputs/journeys_train.csv` if the app expects them. Docker does not automatically generate these files.
3. **Build context:** `.dockerignore` does not exclude `.csv`, `.joblib`, or recommendation output data. Check repository size and data permissions before building/sharing an image.
4. **Health probes:** the Kubernetes manifests use Streamlit's `/_stcore/health` endpoint. Verify it responds with your Streamlit version; adjust if needed.
5. **Resources:** CPU and memory settings are starter estimates. Test actual usage before changing them or increasing replicas.
6. **Secrets:** do not put passwords, API keys, or private credentials in the image or Git repository.

## Suggested Git handover

Work on the `feature/docker-kubernetes` branch. Review the changes:

```powershell
git status
git diff --check
```

After your group reviews the files, commit and push:

```powershell
git add Dockerfile .dockerignore docker-compose.yml k8s/ README_HANDOVER.md
git commit -m "Add Docker and Kubernetes deployment configuration"
git push -u origin feature/docker-kubernetes
```

Only push after testing/review and following your group's contribution process.
