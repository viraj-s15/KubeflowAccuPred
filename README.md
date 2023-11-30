<h3 align="center">Customer Frequency Prediction Pipeline</h3>

<div align="center">

</div>

---

<p align="center"> Completely trains an XGBoost model for Customer Frequency prediction using Kubernetes
    <br> 
</p>

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About ](#-about-)
- [🏁 Getting Started ](#-getting-started-)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
- [🔧 Running the trainging Pipeline ](#-running-the-trainging-pipeline-)
- [🎈 Monitoring ](#-monitoring-)
- [🚀 Deployment ](#-deployment-)
- [⛏️ Built Using ](#️-built-using-)
- [✍️ Authors ](#️-authors-)

## 🧐 About <a name = "about"></a>

Components in the pipeline include:
- Data prerocessing
- Data Splitting
- Hyperparameter Optimisation and Cross Validation
- Model Saving (Local or Cloud)
- Model testing on new data

## 🏁 Getting Started <a name = "getting_started"></a>

You will need k8s set up with kubeflow before doing anything.

### Prerequisites

Make sure you have poetry installed, if you use arch you can install it by running:
```
yay -S python-poetry 
```

This is how I setup Kubeflow with MiniKube

I have kubectl aliased to `minikube kubectl --`

```
minikube start --cpus 4 --memory 4000 --disk-size=10g


curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh"  | bash

git clone git@github.com:kubeflow/manifests.git

cd manifests

while ! kustomize build example | awk '!/well-defined/' | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done

kubectl get pods -A
# To make sure its working
# Give all the pods some time to get running

kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

### Installing

To install all dependencies, I highly recommend using a virtual environment


```
poetry install
```

## 🔧 Running the trainging Pipeline <a name = "tests"></a>

```
poetry run python train_pipeline_execute.py --host http://your_kubeflow_host_url
```

## 🎈 Monitoring <a name="usage"></a>

This project uses Aim for monitoring model performance. 
Run `aim up` in your terminal after the completion of the pipeline.
This will open a dashboard in your browser for monitoring.

It will look something like this

## 🚀 Deployment <a name = "deployment"></a>

Gradio frontend has been built, can be deployed anywhere using the `inference.py` script

## ⛏️ Built Using <a name = "built_using"></a>

- Python
- XGBoost
- Aim - Monitoring and Tracking
- Kubeflow - Distributed computing for training and deployment
- Gradio - API

## ✍️ Authors <a name = "authors"></a>

- [@viraj-s15](https://github.com/viraj-s15) 