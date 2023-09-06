# Machine Learning in Computer Vision

## DBFace

DBFace is a real-time, single-stage detector for face detection, with faster speed and higher accuracy


## Install library from github

The easiest way to install the library is to use `pip`.

```bash
pip install git+https://github.com/Natalya7oct/dbface.git
```

Now you can use library dbface_lib.


## Install dependencies from wheel

You can intsall all dependencies from wheel.

```bash
pip install dist/dbface_package-0.1.0-py3-none-any.whl
```

## Install dependencies from wheel

You need to activate virtual environment in your folder using the command below. It helps you to segregate the project from your global environment.

```bash
python3 -m venv .venv
```

## Web-demo

The project supports web-page demo.

To build an image you may want to run the command similar to the command below:

```bash
docker build -t web_demo .
```
Docker will use [default Dockerfile](./Dockerfile) to build an image. To run the container you need to specify image and port.

```bash
docker run -p 8000:8000 web_demo
```

Open the app using `localhost:8000`.


### Description

Web app allows you to pick up an video from you local files and run demo on it.

[Local image](./dbface_package/demo_scrin.jpg)