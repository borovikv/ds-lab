include .env

current_dir = $(shell pwd)


build:
	touch requirements.dev.txt
	cp ../jupiter-projects/requirements.txt requirements.dev.txt 2>/dev/null || :
	docker-compose -f docker-compose.yml build

# starting services
up:
	docker-compose -f docker-compose.yml up -d

down:
	docker-compose -f docker-compose.yml down -v

project:
	python manage.py $(name)

jupyter:
	open $(shell docker exec -it ds-lab  jupyter server list --json | python3 -c "import sys, json; print('http://127.0.0.1:8888/?token=' + json.load(sys.stdin)['token'])")

