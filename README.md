# Yelp-Data-Exploration

The code is already wrapped in the Docker Image. If you already have Docker installed on your computer, follow the steps below to download the Docker Image and build your own Container to run this code.

### How to execute code in Docker container

1. pull image from docker hub
`docker pull shayan113/yelp`

2. check the image and get ImageID
`docker images`

3. build a container from image,
`docker run -it ImageID /bin/bash`

4. in container environment, change working directory to yelp folder
`cd yelp/`

5. two files are stored in `yelp/`   `.py` file is source code, `.csv` file is data
run command `python3.5 Untitled-Copy1.py` to processing data.


6. after running, stop container
`docker stop container_ID`

7. before next running of container, start container first
`docker start container_ID`
then
`docker exec -it container_ID /bin/bash`


