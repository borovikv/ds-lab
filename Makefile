include .env

current_dir = $(shell pwd)


docker-build:
	docker-compose -f docker-compose.yml build

docker-push:
	docker-compose -f docker-compose.yml push

# starting services
docker-up:
	docker-compose -f docker-compose.yml up -d

docker-down:
	docker-compose -f docker-compose.yml down -v

startproject:
	python manage.py start_project $(name)

pull-data:
	python manage.py pull_data $(name)

push-data:
	python manage.py push_data $(name)
