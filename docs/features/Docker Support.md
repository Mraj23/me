Quartz comes shipped with a Docker image that will allow you to preview your Quartz locally without installing Node.

You can run the below one-liner to run Quartz in Docker.

```sh
docker run --rm -itp 8081:8081 -p 3002:3002 -v ./content:/usr/src/app/content $(docker build -q .)
```
