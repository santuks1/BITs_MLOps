#!/bin/bash

# Setup Kubernetes environment for heart-disease-api

set -e

echo "=== Setting up Kubernetes Environment ==="

# Create namespace (optional, using default)
# kubectl create namespace heart-disease

# Create ConfigMap
echo "Creating ConfigMap..."
kubectl apply -f k8s/configmap.yaml

# Create deployment
echo "Creating Deployment..."
kubectl apply -f k8s/deployment.yaml

# Create service
echo "Creating Service..."
kubectl apply -f k8s/service.yaml

# Create HPA
echo "Creating Horizontal Pod Autoscaler..."
kubectl apply -f k8s/hpa.yaml

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/heart-disease-api

# Get service info
echo ""
echo "=== Deployment Complete ==="
echo "Service Status:"
kubectl get svc heart-disease-api
echo ""
echo "Pod Status:"
kubectl get pods -l app=heart-disease-api
echo ""
echo "To get the external IP:"
echo "kubectl get svc heart-disease-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
