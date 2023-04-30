


docker build -t teststripdx:0.0.1 .
docker run --rm -t -i --mount "type=bind,src=/scratch,dst=/scratch" teststripdx:0.0.1

# Run command
docker run --rm -t -i -v /scratch:/scratch teststripdx:0.0.1

# Interactive
docker run --rm -t -i --entrypoint /bin/bash -v /scratch:/scratch teststripdx:0.0.1

docker tag dc254e4b201c timothystephens/teststripdx:latest
docker tag dc254e4b201c timothystephens/teststripdx:v0.0.1
docker tag dc254e4b201c timothystephens/teststripdx:v0.0
docker tag dc254e4b201c timothystephens/teststripdx:v0

docker push --all-tags timothystephens/teststripdx

singularity pull docker://timothystephens/teststripdx:0.0.1





