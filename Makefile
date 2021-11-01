include .env

current_dir = $(shell pwd)


build:
	docker-compose -f docker-compose.yml build

push:
	docker-compose -f docker-compose.yml push

# starting services
up:
	docker-compose -f docker-compose.yml up -d

down:
	docker-compose -f docker-compose.yml down -v

startproject:
	python manage.py $(name)

url:
	docker exec -it ds-lab  jupyter server list --json | python3 -c "import sys, json; print('http://127.0.0.1:8888/?token=' + json.load(sys.stdin)['token'])"
