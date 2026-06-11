"""
Deployment and production utilities for the churn prediction system
"""
import docker
import yaml
import json
import os
from datetime import datetime
import subprocess
import sys

class DeploymentManager:
    def __init__(self):
        self.deployment_configs = {}
        self.active_deployments = {}
    
    def create_docker_config(self, app_name="churn-predictor", port=7863):
        """Create Docker configuration files"""
        
        # Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p reports utils

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/ || exit 1

# Run application
CMD ["python", "app_advanced.py"]
"""
        
        # Docker Compose
        compose_content = f"""
version: '3.8'

services:
  {app_name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - PYTHONPATH=/app
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - churn-network

  redis:
    image: redis:alpine
    restart: unless-stopped
    networks:
      - churn-network
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - {app_name}
    restart: unless-stopped
    networks:
      - churn-network

volumes:
  redis_data:

networks:
  churn-network:
    driver: bridge
"""
        
        # Nginx configuration
        nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream app {{
        server {app_name}:{port};
    }}

    server {{
        listen 80;
        server_name localhost;

        location / {{
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        location /health {{
            proxy_pass http://app/;
            access_log off;
        }}
    }}
}}
"""
        
        # Write files
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        
        with open('nginx.conf', 'w') as f:
            f.write(nginx_config)
        
        return {
            'dockerfile': 'Dockerfile',
            'compose': 'docker-compose.yml',
            'nginx': 'nginx.conf'
        }
    
    def create_kubernetes_config(self, app_name="churn-predictor", namespace="default"):
        """Create Kubernetes deployment configuration"""
        
        k8s_config = {
            'deployment': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': f'{app_name}-deployment',
                    'namespace': namespace,
                    'labels': {
                        'app': app_name
                    }
                },
                'spec': {
                    'replicas': 3,
                    'selector': {
                        'matchLabels': {
                            'app': app_name
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': app_name
                            }
                        },
                        'spec': {
                            'containers': [
                                {
                                    'name': app_name,
                                    'image': f'{app_name}:latest',
                                    'ports': [
                                        {
                                            'containerPort': 7863
                                        }
                                    ],
                                    'env': [
                                        {
                                            'name': 'ENVIRONMENT',
                                            'value': 'production'
                                        }
                                    ],
                                    'resources': {
                                        'requests': {
                                            'memory': '512Mi',
                                            'cpu': '250m'
                                        },
                                        'limits': {
                                            'memory': '1Gi',
                                            'cpu': '500m'
                                        }
                                    },
                                    'livenessProbe': {
                                        'httpGet': {
                                            'path': '/',
                                            'port': 7863
                                        },
                                        'initialDelaySeconds': 30,
                                        'periodSeconds': 10
                                    },
                                    'readinessProbe': {
                                        'httpGet': {
                                            'path': '/',
                                            'port': 7863
                                        },
                                        'initialDelaySeconds': 5,
                                        'periodSeconds': 5
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            'service': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': f'{app_name}-service',
                    'namespace': namespace
                },
                'spec': {
                    'selector': {
                        'app': app_name
                    },
                    'ports': [
                        {
                            'protocol': 'TCP',
                            'port': 80,
                            'targetPort': 7863
                        }
                    ],
                    'type': 'LoadBalancer'
                }
            },
            'hpa': {
                'apiVersion': 'autoscaling/v2',
                'kind': 'HorizontalPodAutoscaler',
                'metadata': {
                    'name': f'{app_name}-hpa',
                    'namespace': namespace
                },
                'spec': {
                    'scaleTargetRef': {
                        'apiVersion': 'apps/v1',
                        'kind': 'Deployment',
                        'name': f'{app_name}-deployment'
                    },
                    'minReplicas': 2,
                    'maxReplicas': 10,
                    'metrics': [
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'cpu',
                                'target': {
                                    'type': 'Utilization',
                                    'averageUtilization': 70
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # Write Kubernetes YAML files
        for resource_type, config in k8s_config.items():
            filename = f'{app_name}-{resource_type}.yaml'
            with open(filename, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return k8s_config
    
    def create_terraform_config(self, cloud_provider="aws"):
        """Create Terraform infrastructure configuration"""
        
        if cloud_provider == "aws":
            terraform_config = """
# AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  default     = "production"
}

# VPC
resource "aws_vpc" "churn_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "churn-predictor-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "churn_igw" {
  vpc_id = aws_vpc.churn_vpc.id

  tags = {
    Name = "churn-predictor-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public_subnet" {
  count             = 2
  vpc_id            = aws_vpc.churn_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "churn-predictor-public-${count.index + 1}"
  }
}

# Route Table
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.churn_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.churn_igw.id
  }

  tags = {
    Name = "churn-predictor-public-rt"
  }
}

# Route Table Association
resource "aws_route_table_association" "public_rta" {
  count          = length(aws_subnet.public_subnet)
  subnet_id      = aws_subnet.public_subnet[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

# Security Group
resource "aws_security_group" "churn_sg" {
  name_prefix = "churn-predictor"
  vpc_id      = aws_vpc.churn_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "churn-predictor-sg"
  }
}

# Load Balancer
resource "aws_lb" "churn_alb" {
  name               = "churn-predictor-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.churn_sg.id]
  subnets            = aws_subnet.public_subnet[*].id

  enable_deletion_protection = false

  tags = {
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "churn_cluster" {
  name = "churn-predictor"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Data source
data "aws_availability_zones" "available" {
  state = "available"
}

# Outputs
output "load_balancer_dns" {
  value = aws_lb.churn_alb.dns_name
}

output "vpc_id" {
  value = aws_vpc.churn_vpc.id
}
"""
            
            with open('main.tf', 'w') as f:
                f.write(terraform_config)
        
        return terraform_config
    
    def deploy_to_cloud(self, platform="docker", config=None):
        """Deploy application to specified cloud platform"""
        
        deployment_id = f"deploy_{int(datetime.now().timestamp())}"
        
        if platform == "docker":
            return self._deploy_docker(deployment_id, config)
        elif platform == "kubernetes":
            return self._deploy_kubernetes(deployment_id, config)
        elif platform == "aws":
            return self._deploy_aws(deployment_id, config)
        else:
            return {"success": False, "error": f"Platform {platform} not supported"}
    
    def _deploy_docker(self, deployment_id, config):
        """Deploy using Docker Compose"""
        try:
            # Create Docker configs
            docker_files = self.create_docker_config()
            
            # Build and deploy
            result = subprocess.run(
                ["docker-compose", "up", "-d", "--build"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                self.active_deployments[deployment_id] = {
                    'platform': 'docker',
                    'status': 'running',
                    'deployed_at': datetime.now(),
                    'config': docker_files
                }
                
                return {
                    "success": True,
                    "deployment_id": deployment_id,
                    "message": "Docker deployment successful",
                    "logs": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _deploy_kubernetes(self, deployment_id, config):
        """Deploy to Kubernetes cluster"""
        try:
            # Create K8s configs
            k8s_configs = self.create_kubernetes_config()
            
            # Apply configurations
            for resource_type in k8s_configs:
                filename = f'churn-predictor-{resource_type}.yaml'
                result = subprocess.run(
                    ["kubectl", "apply", "-f", filename],
                    capture_output=True, text=True
                )
                
                if result.returncode != 0:
                    return {"success": False, "error": result.stderr}
            
            self.active_deployments[deployment_id] = {
                'platform': 'kubernetes',
                'status': 'running',
                'deployed_at': datetime.now(),
                'config': k8s_configs
            }
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "message": "Kubernetes deployment successful"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _deploy_aws(self, deployment_id, config):
        """Deploy to AWS using Terraform"""
        try:
            # Create Terraform config
            self.create_terraform_config("aws")
            
            # Initialize and apply Terraform
            commands = [
                ["terraform", "init"],
                ["terraform", "plan"],
                ["terraform", "apply", "-auto-approve"]
            ]
            
            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "error": result.stderr}
            
            self.active_deployments[deployment_id] = {
                'platform': 'aws',
                'status': 'running',
                'deployed_at': datetime.now(),
                'terraform_applied': True
            }
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "message": "AWS deployment successful"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_deployment_status(self, deployment_id=None):
        """Get status of deployments"""
        if deployment_id:
            return self.active_deployments.get(deployment_id, {"error": "Deployment not found"})
        return self.active_deployments
    
    def create_ci_cd_pipeline(self):
        """Create GitHub Actions CI/CD pipeline"""
        
        github_workflow = """
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python -m pytest tests/ -v
        
    - name: Run linting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t churn-predictor:${{ github.sha }} .
        
    - name: Log in to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
        
    - name: Push to Docker Hub
      run: |
        docker tag churn-predictor:${{ github.sha }} ${{ secrets.DOCKER_USERNAME }}/churn-predictor:latest
        docker push ${{ secrets.DOCKER_USERNAME }}/churn-predictor:latest
        
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Add actual deployment commands here
"""
        
        # Create .github/workflows directory
        os.makedirs('.github/workflows', exist_ok=True)
        
        with open('.github/workflows/ci-cd.yml', 'w') as f:
            f.write(github_workflow)
        
        return github_workflow

# Global deployment manager
deployment_manager = DeploymentManager()