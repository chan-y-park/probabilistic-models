# Docker-related files

## Quickstart
```
$ docker build -t tfp tfp
$ docker-compose run tfp 
```

## nvidia-docker
```
$ docker run --runtime=nvidia -it tfp
$ NV_GPU=1 nvidia-docker run -ti tfp
```
