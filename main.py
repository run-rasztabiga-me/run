from fastapi import FastAPI

app = FastAPI()


async def deploy():
    # clone repo (shallow)
    # find dockerfile
    # build docker image
    # push docker image to cluster?
    # create k8s deployment + service + ingress config giles
    # apply k8s config files
    # return url

    pass


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
