# probabilistic-models
A collection of probabilistic models (actually a sandbox for now)

# tf-probability docker image 
## Building the docker image 
```
$ docker build -t tfp tfp-docker-image
```
## Running the docker image
```
$ docker run --runtime=nvidia -it tfp
```
### GPU isolation
```
$ NV_GPU=1 nvidia-docker run -ti tfp
```

# Running docker-compose 
```
$ docker-compose run --service-ports tfp 
```
