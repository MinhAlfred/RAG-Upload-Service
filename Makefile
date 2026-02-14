# Docker Makefile for OCR Embedding Service
.PHONY: help build up down dev prod logs clean test health

# Variables
COMPOSE_FILE = docker-compose.yml
SERVICE_NAME = ocr-embedding-service

help: ## Show this help message
	@echo "OCR Embedding Service - Docker Commands"
	@echo "======================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development Commands

dev: ## Start in development mode with hot reload
	docker-compose up --build

dev-bg: ## Start in development mode (background)
	docker-compose up --build -d

local: ## Start with local Qdrant database
	docker-compose --profile local-qdrant up --build

##@ Production Commands

prod: ## Start in production mode  
	docker-compose -f $(COMPOSE_FILE) up --build -d

prod-scale: ## Start production with 3 replicas
	docker-compose -f $(COMPOSE_FILE) up --scale $(SERVICE_NAME)=3 --build -d

##@ Basic Operations

build: ## Build the Docker image
	docker-compose build

up: ## Start services
	docker-compose up -d

down: ## Stop and remove services
	docker-compose down

restart: ## Restart services
	docker-compose restart

##@ Monitoring & Debugging

logs: ## View logs
	docker-compose logs -f $(SERVICE_NAME)

logs-all: ## View all service logs
	docker-compose logs -f

health: ## Check service health
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Service not responding"

shell: ## Access container shell
	docker-compose exec $(SERVICE_NAME) bash

##@ Maintenance

clean: ## Stop and clean everything (destructive)
	docker-compose down -v --rmi all
	docker system prune -f

clean-volumes: ## Remove all volumes (destructive)
	docker-compose down -v

setup: ## Setup environment file
	cp .env.example .env
	@echo "Created .env file. Please edit with your credentials."

##@ Testing

test: ## Run tests inside container
	docker-compose exec $(SERVICE_NAME) python -m pytest

test-api: ## Test API endpoints
	@echo "Testing extract endpoint..."
	@curl -X GET http://localhost:8000/health
	@echo "\nTesting health endpoint..."
	@curl -X GET http://localhost:8000/docs

##@ Information

status: ## Show container status
	docker-compose ps

images: ## Show Docker images
	docker images | grep ocr

stats: ## Show container resource usage
	docker stats

##@ Quick Setup

install: setup build ## Full setup: copy env + build
	@echo "Setup complete! Edit .env file and run 'make dev'"

quick-start: ## Quick start for first time users
	@echo "Quick Start Guide:"
	@echo "1. make setup     # Creates .env file"  
	@echo "2. Edit .env with your API keys"
	@echo "3. make dev       # Start development server"
	@echo "4. Open http://localhost:8000/docs"

##@ Docker Commands

docker-build: ## Build with plain Docker
	docker build -t ocr-embedding-service .

docker-run: ## Run with plain Docker
	docker run -p 8000:8000 --env-file .env ocr-embedding-service

##@ Environment Commands

env-check: ## Check environment variables
	@echo "Checking .env file..."
	@test -f .env && echo "✓ .env exists" || echo "✗ .env missing (run 'make setup')"
	@grep -q "OPENAI_API_KEY=" .env 2>/dev/null && echo "✓ OPENAI_API_KEY set" || echo "✗ OPENAI_API_KEY missing"
	@grep -q "QDRANT" .env 2>/dev/null && echo "✓ Qdrant config found" || echo "✗ Qdrant config missing"