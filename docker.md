
Tensorflow 2.0

    sudo docker run --name tf2.0 -it -v [LOCAL_PATH]:/project/nonlinear tensorflow/tensorflow:2.0.0b1-gpu-py3 bash
    sudo nvidia-docker run --name tf2.0rc0-gpu -it -v /mnt/mainblob/nonlinearML:/project/nonlinear tensorflow/tensorflow:2.0.0rc0-gpu-py3 bash

Attaching to container

    sudo docker start CONTAINER_ID
    sudo docker attach CONTAINER_ID

