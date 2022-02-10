docker run --rm --ipc=host --gpus all -it \
-v $(pwd)/workspace:/usr/src/app/workspace \
damage_detect bash -c 'bash $workspace/detect.sh'