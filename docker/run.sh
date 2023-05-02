

TAG="dev-StaticCoords_v0.0.3"

USER="timothystephens"
PROG="teststripdx"



## Build docker file from instructions
#  use '--no-cache' to ensure the github repo is always re-downloaded when building
cp -r ../models .
docker build --no-cache -t $PROG:$TAG . 2>&1 | tee build.log
rm -r models



## Build docker image and push ro repo
BUILD=$(awk '$1=="Successfully" && $2=="built" {print $3}' build.log)

docker tag $BUILD $USER/$PROG:${TAG}
docker tag $BUILD $USER/$PROG:${TAG%.*}
docker tag $BUILD $USER/$PROG:${TAG%%.*}
#docker tag $BUILD $USER/teststripdx:latest

docker push --all-tags $USER/$PROG



## Test pull using singularity
conda activate singularity
IMG="${PROG}:${TAG%%.*}"
singularity pull docker://$USER/$IMG

IMG="${PROG}_${TAG%%.*}"
singularity run ${IMG}.sif process  -i ../test/vid*.mp4
singularity run ${IMG}.sif combine  -i ../test/vid*.results.txt   -o ../test/combined.results.txt
singularity run ${IMG}.sif joinPDFs -i ../test/vid*.detection.pdf -o ../test/combined.detection.pdf



## Test with docker
#    - probabily wont work since docker is running as 
#      root, and this means that we cant create file in the users 
#      directiry as we are the wrong user
docker run --workdir=$PWD ${IMG} process  -i ../test/vid*.mp4
docker run --workdir=$PWD ${IMG} combine  -i ../test/vid*.results.txt   -o ../test/combined.results.txt
docker run --workdir=$PWD ${IMG} joinPDFs -i ../test/vid*.detection.pdf -o ../test/combined.detection.pdf



## Cleanup docker, singularity, and test data
docker container prune -a
docker image prune -af
rm -fr *.sif
rm -fr ../test/*TestStripDX* ../test/combined.*



