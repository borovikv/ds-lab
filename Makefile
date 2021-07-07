include .env

current_dir = $(shell pwd)


docker-build:
	docker-compose -f docker-compose.yml build

docker-push:
	docker-compose -f docker-compose.yml push

# starting services
start-service:
	docker-compose -f docker-compose.yml up -d

stop-service:
	docker-compose -f docker-compose.yml down -v
